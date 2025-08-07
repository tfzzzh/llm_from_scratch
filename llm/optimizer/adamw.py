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
        cpu_off_loading: bool = False
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
        self.cpu_off_loading = cpu_off_loading

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # get device for parameter and device for state
                device_param = p.device
                device_state = device_param if not self.cpu_off_loading else torch.device("cpu")

                # get states
                grad = p.grad.data
                grad = grad.to(torch.float32).to(device_state)

                state = self.state[p]
                # initialize state
                # make momentum to float32 for accuracy requirement
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32, device=device_state)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32, device=device_state)

                t = state["t"]
                m = state["m"]
                v = state["v"]

                # compute momentum & curvature
                m += (1.0 - self.betas[0]) * (grad - m)
                v += (1.0 - self.betas[1]) * (grad.square() - v)

                # compute lr at t
                lr_t = (
                    lr
                    * math.sqrt(1.0 - math.pow(self.betas[1], t + 1))
                    / (1.0 - math.pow(self.betas[0], t + 1))
                )

                # update parameters
                x = p.data.to(torch.float32)
                x -= lr_t * (m.to(device_param) / (torch.sqrt(v.to(device_param)) + self.eps))
                x -= lr * self.weight_decay * x
                p.data = x.to(p.data.dtype)

                # update states
                state["t"] = t + 1

        return loss