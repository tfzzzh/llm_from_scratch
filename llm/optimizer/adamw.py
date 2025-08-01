from typing import Optional, Callable, Iterable, Tuple
import torch
from torch import nn
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
        eps: float,
    ):
        # check valid
        assert weight_decay >= 0
        assert betas[0] > 0 and betas[0] < 1
        assert betas[1] > 0 and betas[1] < 1
        assert eps >= 0

        super().__init__(params, dict(lr=lr))
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # get states
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 0)
                #   make momentum to float32 for accuracy requirement
                grad = grad.to(torch.float32)
                m = state.get("m", torch.zeros_like(p, dtype=torch.float32))
                v = state.get("v", torch.zeros_like(p, dtype=torch.float32))

                # compute momentum & curvature
                m = self.betas[0] * m + (1 - self.betas[0]) * grad
                v = self.betas[1] * v + (1 - self.betas[1]) * grad * grad

                # compute lr at t
                lr_t = (
                    lr
                    * math.sqrt(1.0 - math.pow(self.betas[1], t + 1))
                    / (1.0 - math.pow(self.betas[0], t + 1))
                )

                # update parameters
                x = p.data.to(torch.float32)
                x -= lr_t * (m / (torch.sqrt(v) + self.eps))
                x -= lr * self.weight_decay * x
                p.data = x.to(p.data.dtype)

                # update states
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss