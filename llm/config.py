import yaml
import numpy as np
import torch
import copy
from .tokenizer import Tokenizer
from .utility import DataLoader
from .layers import TransformerLM
from .optimizer import AdamW
from .lr_scheduler import CosineAnnealLR
from .data_parallel_wrap import DataParallelBucket, DataParallelReduceInterleaved


def read_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def get_device(device: str):
    return torch.device(device=device)


def load_tokenizer(config):
    tokenizer = Tokenizer.from_pickle(config["tokenizer"]["tokenizer_path"])
    assert len(tokenizer.vocab) == config["model"]["vocab_size"]


def make_train_dataloader(config):
    if config['data']['type'] == 'random':
        data_len = config['data']['dataset_size']
        num_vocab = config['model']['vocab_size']
        dataset = np.random.randint(0, num_vocab, size=data_len, dtype=np.int64)

    elif config['data']['type'] == 'numpy_data':
        dataset = np.load(config["data"]["train_data_path"], mmap_mode="r")
        
    else:
        raise NotImplementedError
    
    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        context_length=config["model"]["context_length"],
        device=config["device"],
    )
    return loader


def make_model(config):
    model_config = {**config["model"]}
    model_config["device"] = config["device"]

    dtype = _resolve_dtype(model_config['dtype'])
    model_config["dtype"] = dtype
    model = TransformerLM(**model_config)
    model.to(model_config["device"])

    return model

def _resolve_dtype(dtype):
    if dtype == "float32":
        dtype = torch.float32

    elif dtype == "float16":
        dtype = torch.float16

    else:
        raise NotImplementedError
    
    return dtype


def make_optimizer(params, config):
    opt_config = config["optimizer"]
    optimizer = AdamW(
        params,
        lr=opt_config["lr"],
        weight_decay=opt_config["weight_decay"],
        betas=tuple(opt_config["betas"]),
        eps=opt_config["eps"],
    )
    return optimizer


def make_schedule(optimizer, config):
    scheduler_config = copy.deepcopy(config["scheduler"])
    assert scheduler_config["type"] == "cosine_annealing"
    scheduler = CosineAnnealLR(
        optimizer,
        lr_min=scheduler_config["lr_min"],
        lr_max=scheduler_config["lr_max"],
        warm_up_steps=scheduler_config["warm_up_steps"],
        cosin_ann_steps=scheduler_config["cosin_ann_steps"],
    )
    return scheduler


def make_dp_wrap(model, config):
    model_dtype = _resolve_dtype(config['model']['dtype'])
    wrap_type = config['dp_wrap']['type']
    if wrap_type == 'DataParallelBucket':
        bucket_size_mb = config['dp_wrap']['bucket_size_mb']
        grad_type = _resolve_dtype(config['dp_wrap']['grad_type'])
        return DataParallelBucket(model, model_dtype, None, bucket_size_mb, grad_type)

    elif wrap_type == 'DataParallelReduceInterleaved':
        return DataParallelReduceInterleaved(model, model_dtype, None)

    else:
        raise NotImplementedError