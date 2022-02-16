import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from flearn.users.user_base import User
from flearn.optimizers.fedoptimizer import *
from flearn.differential_privacy.differential_privacy import *
from torch.optim.lr_scheduler import StepLR
from utils.autograd_hacks import *


# Implementation for FedAvg users

class UserAVG(User):
    def __init__(self, numeric_id, train_data, test_data, model, sample_ratio, learning_rate, L, local_updates,
                 dp, times, use_cuda):
        super().__init__(numeric_id, train_data, test_data, model[0], sample_ratio, learning_rate, L,
                         local_updates, dp, times, use_cuda)

        if model[1] == 'mclr':
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
            # self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.1)
            # self.lr_drop_rate = 0.95

        param_groups = [{'params': p, 'lr': self.learning_rate} for p in self.model.parameters()]
        self.optimizer = FedAvgOptimizer(param_groups, lr=self.learning_rate, weight_decay=L)
        self.csi = None

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train_no_dp(self, glob_iter):
        """Training phase without differential privacy"""
        for epoch in range(1, self.local_updates + 1):
            self.model.train()

            # new batch (data sampling on every local epoch)
            np.random.seed(500 * (self.times + 1) * (glob_iter + 1) + epoch + 1)
            torch.manual_seed(500 * (self.times + 1) * (glob_iter + 1) + epoch + 1)
            train_idx = np.arange(self.train_samples)
            train_sampler = SubsetRandomSampler(train_idx)
            self.trainloader = DataLoader(self.train_data, self.batch_size, sampler=train_sampler)

            X, y = list(self.trainloader)[0]

            if self.use_cuda:
                X, y = X.cuda(), y.cuda()

            self.optimizer.zero_grad()
            clear_backprops(self.model)
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()

            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

        # get model difference
        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()

        return loss

    def train_dp(self, sigma_g, glob_iter, max_norm):
        """Training phase under differential privacy"""

        for epoch in range(1, self.local_updates + 1):
            self.model.train()

            # new batch (data sampling on every local epoch)
            np.random.seed(500 * (self.times + 1) * (glob_iter + 1) + epoch + 1)
            torch.manual_seed(500 * (self.times + 1) * (glob_iter + 1) + epoch + 1)
            train_idx = np.arange(self.train_samples)
            train_sampler = SubsetRandomSampler(train_idx)
            self.trainloader = DataLoader(self.train_data, self.batch_size, sampler=train_sampler)

            X, y = list(self.trainloader)[0]

            if self.use_cuda:
                X, y = X.cuda(), y.cuda()

            self.optimizer.zero_grad()
            clear_backprops(self.model)
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward(retain_graph=True)
            compute_grad1(self.model)

            for p in self.model.parameters():
                # clipping single gradients

                # heuristic: otherwise, use max_norm constant
                max_norm = np.median([float(grad.data.norm(2)) for grad in p.grad1])

                p.grad1 = torch.stack(
                    [grad / max(1, float(grad.data.norm(2)) / max_norm) for grad in p.grad1])
                p.grad.data = torch.mean(p.grad1, dim=0)
                # DP mechanism
                p.grad.data = GaussianMechanism(p.grad.data, sigma_g, max_norm, self.batch_size, self.use_cuda)

            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

        # get model difference
        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()

        return 0

    def get_params_norm(self):
        """Returns ||x_user^t+1 -x_server^t||."""
        params = []
        for delta in self.delta_model:
            params.append(torch.flatten(delta.data))
        return float(torch.norm(torch.cat(params)))
