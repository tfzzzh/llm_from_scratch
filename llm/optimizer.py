from collections.abc import Callable, Iterable
from typing import Optional, Tuple
import torch
from torch import nn
import math

def gradient_clipping_(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    """clip graident inplace
        if |grad| <= max_l2_norm leave it untacted
        else set grad = grad / (|grad| + e) * max_l2_norm

    note: gradient_clipping use total gradnorm instead of using gradient for a single variable
    """
    total_norm = 0.0
    for param in parameters:
        if param.grad is None:
            continue

        gradnorm = torch.norm(param.grad.data)
        total_norm += gradnorm * gradnorm

    total_norm = math.sqrt(total_norm)
    if total_norm > max_l2_norm:
        for param in parameters:
            if param.grad is None:
                continue

            param.grad.data = param.grad.data / (total_norm + eps) * max_l2_norm

class SGD(torch.optim.Optimizer):
    """
    How pytorch group parameters
    1. single group
    '''python
    # All parameters get the same settings
    optimizer = SGD(model.parameters(), lr=0.01)
    # Creates: self.param_groups = [{'params': [list_of_all_params], 'lr': 0.01}]
    '''

    2. mutiple groups (manul)
    '''python
    optimizer = SGD([
        {'params': model.layer1.parameters(), 'lr': 0.001},
        {'params': model.layer2.parameters(), 'lr': 0.01},
        {'params': model.layer3.parameters()}  # Uses default lr from defaults
    ], lr=0.1)  # default
    # Creates: self.param_groups = [
    #     {'params': [layer1_params], 'lr': 0.001},
    #     {'params': [layer2_params], 'lr': 0.01},
    #     {'params': [layer3_params], 'lr': 0.1}
    # ]
    '''
    Through grouping, you can use different learning rates for different groups
    """

    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        super(SGD, self).__init__(params, dict(lr=lr))

    def step(self, closure: Optional[Callable] = None):
        """
        purpose of closure:
            Re-computes the forward pass and loss
            Re-computes gradients
            Returns the loss value

        some algorithm using linesearch or use multi-step graidents like
            LBFGS need to call the function

        Args:
            closure (Optional[Callable], optional): _description_. Defaults to None.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # when you have a scheduler, the learning rate will be adjust by it
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1  # Increment iteration number.

        return loss


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


# [TODO] add other optimizers