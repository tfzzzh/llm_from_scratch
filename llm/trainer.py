from torch import nn
import torch.nn.functional as F
import tqdm
import os

from .config import (
    get_device,
    load_tokenizer,
    make_train_dataloader,
    make_model,
    make_optimizer,
    make_schedule
)

from .utility import save_checkpoint
from .logger import Logger


class Trainer:
    def __init__(self, config):
        config["device"] = get_device(config["device"])
        print(config)

        self.model = make_model(config)
        self.model.to(config["device"])

        self.tokenizer = load_tokenizer(config)
        self.optimizer = make_optimizer(self.model.parameters(), config)
        self.scheduler = make_schedule(self.optimizer, config)
        self.train_dataloader = make_train_dataloader(config)

        # parameter from training
        self.gradient_clip_norm = config['training']['grad_clip_norm']
        self.max_steps = config['training']['max_steps']
        self.batch_size = config['data']['batch_size']
        
        # logging
        self.logger = Logger(config['logger']['log_dir'])
        self.checkpoint_dir = config['checkpointing']['checkpoint_dir']
        self.checkpoint_per_step = config['checkpointing']['checkpoint_per_step']
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.config = config

    def train(self):
        # open train
        self.model.train()
        for step in tqdm.trange(self.max_steps, dynamic_ncols=True):
            x_train, y_train = self.train_dataloader.get_batch()
            logits = self.model(x_train)

            loss = F.cross_entropy(logits.transpose(-1,-2), y_train)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm)

            self.optimizer.step()
            self.scheduler.step()

            infos = {
                'loss': loss.item(),
                'grad_norm': grad_norm.item(),
                'lr': self.scheduler.get_last_lr()
            }

            self.logger.log_metrics(infos, step)

            if step % self.checkpoint_per_step == 0:
                save_checkpoint(
                    os.path.join(self.checkpoint_dir, f"step={step}.pt"),
                    self.model,
                    self.optimizer,
                    step,
                    self.scheduler
                )
