import torch
import math


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        muon_params,
        adam_params,
        lr_muon=0.02,
        beta_muon=0.95,
        lr_adam=3e-4,
        betas=(0.9, 0.95),
        eps=1e-10,
        ns_steps=5,
        weight_decay=0.0,
    ):
        groups = []
        # append parameter optimized by muon steps
        groups.append(
            {
                "muon_step": True,
                "lr": lr_muon,
                "beta": beta_muon,
                "weight_decay": weight_decay,
                "ns_steps": ns_steps,
                "params": muon_params,
            }
        )

        # append parameter optimized by adam
        groups.append(
            {
                "muon_step": False,
                "lr": lr_adam,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "params": adam_params,
            }
        )

        super().__init__(groups, dict())

    def _muon_step(self, group):
        beta = group["beta"]
        lr = group["lr"]
        gamma = group["weight_decay"]
        ns_steps = group["ns_steps"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad.data
            state = self.state[p]
            if "momentum" not in state:
                state["momentum"] = torch.zeros(p.shape, device=p.device, dtype=p.dtype)

            momentum = state["momentum"]
            momentum += (1.0 - beta) * (grad - momentum)
            delta = (1.0 - beta) * grad + beta * momentum
            delta = zeropower_via_newtonschulz5(delta, steps=ns_steps)
            # [TODO]
            delta *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5

            p.data -= lr * (delta + gamma * p.data)

    def _adam_step(self, group):
        betas = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad.data
            state = self.state[p]

            # init state
            if "t" not in state:
                state["t"] = 0
                state["m"] = torch.zeros(p.shape, device=p.device, dtype=torch.float32)
                state["v"] = torch.zeros(p.shape, device=p.device, dtype=torch.float32)

            # m, v update inplace
            t, m, v = state["t"], state["m"], state["v"]
            m += (1.0 - betas[0]) * (grad - m)
            v += (1.0 - betas[1]) * (grad.square() - v)

            lr_t = (
                lr
                * math.sqrt(1.0 - math.pow(betas[1], t + 1))
                / (1.0 - math.pow(betas[0], t + 1))
            )

            # update parameters
            p.data -= lr_t * (m / (torch.sqrt(v) + eps))
            p.data -= lr * weight_decay * p.data

            state["t"] += 1

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            if group["muon_step"]:
                self._muon_step(group)

            else:
                self._adam_step(group)

        return loss


def zeropower_via_newtonschulz5(G, steps: int, use_inner_bf16=False):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    When performing 10 steps, the distance between U @ V.T and the output matrix is about 0.24 when measured by relative norm distance
        |out - U@V.T| / |U @ V.T|
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if use_inner_bf16 else G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
