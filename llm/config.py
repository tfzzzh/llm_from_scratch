import os
import yaml
import numpy as np
import torch
import copy
import time
from .tokenizer import Tokenizer
from .utility import DataLoader
from .layers import TransformerLM
from .optimizer import AdamW, SGD, Zero, Muon
from .lr_scheduler import CosineAnnealLR, MuonScheduler
from .data_parallel_wrap import DataParallelBucket, DataParallelReduceInterleaved
from .logger import Logger


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


def make_optimizer(params, config, model=None):
    opt_config = config["optimizer"]
    opt_type = config['optimizer']['type']
    if opt_type == 'AdamW':
        optimizer = AdamW(
            params,
            lr=opt_config["lr"],
            weight_decay=opt_config["weight_decay"],
            betas=tuple(opt_config["betas"]),
            eps=opt_config["eps"],
        )

    elif opt_type == "Muon":
        assert model is not None
        mueon_weights = [p for p in model.blocks.parameters() if p.ndim >= 2]
        mueon_weights_set = set(mueon_weights)
        adam_weights = [p for p in model.parameters() if p not in mueon_weights_set]
        optimizer = Muon(
            mueon_weights,
            adam_weights,
            lr_muon=opt_config['lr_muon'],
            beta_muon=opt_config['beta_muon'],
            lr_adam=opt_config['lr_adam'],
            betas=tuple(opt_config['betas']),
            eps=opt_config['eps'],
            ns_steps=opt_config['ns_steps'],
            weight_decay=opt_config['weight_decay']
        )

    elif opt_type == 'Zero':
        inner_opt_config = opt_config['inner_optimizer']
        assert inner_opt_config['type'] == 'AdamW'
        inner_opt_kwargs = {
            'lr': inner_opt_config['lr'],
            'weight_decay': inner_opt_config['weight_decay'],
            'betas': tuple(inner_opt_config['betas']),
            'eps': inner_opt_config['eps']
        }
        optimizer = Zero(
            params, None, AdamW, **inner_opt_kwargs
        )

    else:
        raise NotImplementedError(f"optimizer {opt_type} not implemented")
    
    return optimizer


def make_schedule(optimizer, config):
    scheduler_config = copy.deepcopy(config["scheduler"])
    if scheduler_config["type"] == "CosineAnnealLR":
        scheduler = CosineAnnealLR(
            optimizer,
            lr_min=scheduler_config["lr_min"],
            lr_max=scheduler_config["lr_max"],
            warm_up_steps=scheduler_config["warm_up_steps"],
            cosin_ann_steps=scheduler_config["cosin_ann_steps"],
        )

    elif scheduler_config["type"] == "MuonScheduler":
        scheduler = MuonScheduler(
            optimizer,
            num_iterations=config['training']['max_steps'],
            cooldown_frac=scheduler_config['cooldown_frac']
        )

    else:
        raise NotImplementedError
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
    
def make_tensorboard_logger(config):
    log_dir = config['logger']['log_dir']
    if not (os.path.exists(log_dir)):
        os.makedirs(log_dir)

    log_string = f"{config['base_config']}_opt{config['optimizer']['type']}_dim{config['model']['d_model']}"
    log_string += "_" + time.strftime("%d-%m-%Y_%H-%M-%S")

    logdir = os.path.join(log_dir, log_string)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    return Logger(logdir)