import torch
from torch.optim import Adam
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np


class FedLOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(FedLOptimizer, self).__init__(params, defaults)
        pass


class FedAvgOptimizer(FedLOptimizer):
    def __init__(self, params, lr, weight_decay):
        super().__init__(params, lr, weight_decay)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = p.data - p.grad.data * group['lr']
        return loss


class SCAFFOLDOptimizer(FedLOptimizer):
    def __init__(self, params, lr, weight_decay):
        super().__init__(params, lr, weight_decay)

    def step(self, server_controls, user_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group, c, ci in zip(self.param_groups, server_controls, user_controls):
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data + c.data - ci.data
                p.data = p.data - d_p * group['lr']
        return loss
