from typing import Optional, Callable, Iterable
import torch
from torch import nn
import math


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