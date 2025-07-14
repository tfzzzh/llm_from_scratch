from typing import Optional
import numpy as np
import numpy.typing as npt
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.serialization import FILE_LIKE


class DataLoader:
    def __init__(
        self,
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: Optional[torch.device] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.idx = 0
        self.num_tuples = len(dataset)
        self.num_indices = self.num_tuples - self.context_length  # [0, context_length]
        self.shuffle_indices = np.random.permutation(self.num_indices)

    def get_batch(self):
        if self.idx + self.batch_size > self.num_indices:
            self.idx = 0
            self.shuffle_indices = np.random.permutation(self.num_indices)

        idx = self.idx

        # sentences = np.stack([
        #     self.dataset[
        #         self.shuffle_indices[idx + bid] : self.shuffle_indices[idx + bid] + self.context_length + 1
        #     ]
        #     for bid in range(self.batch_size)],
        #     axis=0
        # )
        start_indices = self.shuffle_indices[idx : idx + self.batch_size]
        linear_indices = start_indices[:, None] + np.arange(
            0, self.context_length + 1, dtype=np.int64
        )
        sentences = self.dataset[linear_indices]

        x = torch.tensor(
            sentences[:, : self.context_length], dtype=torch.int64, device=self.device
        )
        y = torch.tensor(sentences[:, 1:], dtype=torch.int64, device=self.device)

        self.idx += self.batch_size
        return x, y


def save_checkpoint(
    out_path: FILE_LIKE,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    scheduler: Optional[_LRScheduler] = None,
):
    states = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    if scheduler is not None:
        states["scheduler"] = scheduler.state_dict()

    torch.save(states, out_path)


def load_checkpoint(
    src_path: FILE_LIKE,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[_LRScheduler] = None,
):
    states = torch.load(src_path)
    iteration = states['iteration']
    model.load_state_dict(states['model'])
    optimizer.load_state_dict(states['optimizer'])

    if scheduler is not None:
        scheduler.load_state_dict(states['scheduler'])

    return iteration