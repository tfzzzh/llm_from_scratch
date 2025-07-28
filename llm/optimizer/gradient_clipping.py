
"""
This file only examplify how gradient clipping is implemented. I use torch's version in my training code
"""

import torch
from typing import Iterable
import torch
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