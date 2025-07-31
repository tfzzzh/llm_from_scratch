import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
import math


class CosineAnnealLR(_LRScheduler):
    """
    Note that last_epoch set to -1, since initially the optimizer uses the init lr from algorithm itself
    remember the initialization: opt = SGD(param, lr=1e-3)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_min: float,
        lr_max: float,
        warm_up_steps: int,
        cosin_ann_steps: int,
        last_epoch=-1,
    ):
        assert lr_max >= 0 and lr_min >= 0 and lr_min <= lr_max
        assert warm_up_steps >= 0
        assert warm_up_steps <= cosin_ann_steps


        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up_steps = warm_up_steps
        self.cosin_ann_steps = cosin_ann_steps

        # !!! must be here
        super().__init__(optimizer, last_epoch=last_epoch)

    @staticmethod
    def _compute_lr(
        last_epoch, lr_min, lr_max, warm_up_steps, cosin_ann_steps, use_init_zero=False
    ):
        t = last_epoch
        assert t >= 0

        if t < warm_up_steps:
            return (
                (t + 1) / warm_up_steps * lr_max
                if not use_init_zero
                else t / warm_up_steps * lr_max
            )

        if t < cosin_ann_steps:
            return lr_min + 0.5 * (
                1.0
                + math.cos(
                    (t - warm_up_steps) / (cosin_ann_steps - warm_up_steps) * math.pi
                )
            ) * (lr_max - lr_min)

        return lr_min

    def get_lr(self):
        """
        Note that get_lr returns a list for each parameter group, not a single float
        """
        lr = CosineAnnealLR._compute_lr(
            self.last_epoch,
            self.lr_min,
            self.lr_max,
            self.warm_up_steps,
            self.cosin_ann_steps,
        )
        return [lr for _ in self.base_lrs]

    # shall I rewrite state_dict?
    # def state_dict(self):
    #     """Return state of the scheduler"""
    #     state = super().state_dict()
    #     state.update({
    #         'lr_min': self.lr_min,
    #         'lr_max': self.lr_max,
    #         'warm_up_steps': self.warm_up_steps,
    #         'cosin_ann_steps': self.cosin_ann_steps,
    #     })
    #     return state
    
    # def load_state_dict(self, state_dict):
    #     """Load state of the scheduler"""
    #     # Load custom parameters
    #     self.lr_min = state_dict.pop('lr_min')
    #     self.lr_max = state_dict.pop('lr_max')
    #     self.warm_up_steps = state_dict.pop('warm_up_steps')
    #     self.cosin_ann_steps = state_dict.pop('cosin_ann_steps')
        
    #     # Load base scheduler state
    #     super().load_state_dict(state_dict)


class MuonScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, num_iterations:int, cooldown_frac:float = 0.4,last_epoch: int = -1):
        # bookmark initial lr
        self.initial_lrs = []
        for group in optimizer.param_groups:
            self.initial_lrs.append(group['lr'])

        self.num_iterations = num_iterations
        self.cooldown_frac = cooldown_frac

        # !!! must be here
        super().__init__(optimizer, last_epoch)


    @staticmethod
    def _get_lr_scale(step, num_iterations, cooldown_frac):
        x = step / num_iterations
        assert 0.0 <= x <= 1.0

        if x < 1 - cooldown_frac:
            return 1.0
    
        else:
            w = (1 - x) / cooldown_frac
            return w * 1.0 + (1 - w) * 0.1
        
    def get_lr(self):
        scale = MuonScheduler._get_lr_scale(
            self.last_epoch + 1,
            self.num_iterations,
            self.cooldown_frac
        )

        return [lr_init * scale for lr_init in self.initial_lrs]