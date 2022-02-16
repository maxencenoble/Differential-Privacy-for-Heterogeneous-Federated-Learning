import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from flearn.users.user_base import User
from flearn.optimizers.fedoptimizer import *
from flearn.differential_privacy.differential_privacy import *
import math
import copy
from torch.optim.lr_scheduler import StepLR
from utils.autograd_hacks import *


# Implementation for SCAFFOLD users

class UserSCAFFOLD(User):
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
        self.optimizer = SCAFFOLDOptimizer(param_groups, lr=self.learning_rate, weight_decay=L)

        self.controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.server_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.delta_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.csi = None

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def set_first_controls_no_dp(self):
        """Warm start strategy without differential privacy"""
        grads = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]

        for epoch in range(1, self.local_updates + 1):
            self.model.eval()
            self.optimizer.zero_grad()

            # new batch (data sampling on every local epoch)
            np.random.seed(500 * (self.times + 1) + epoch + 1)
            torch.manual_seed(500 * (self.times + 1) + epoch + 1)
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

            for p, grad in zip(self.model.parameters(), grads):
                grad += p.grad.data

        for control, grad in zip(self.controls, grads):
            control.data = grad / self.local_updates

        self.optimizer.zero_grad()

    def set_first_controls_dp(self, sigma_g, max_norm):
        """Warm start strategy under differential privacy"""
        grads = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]

        for epoch in range(1, self.local_updates + 1):
            self.model.eval()
            self.optimizer.zero_grad()

            # new batch (data sampling on every local epoch)
            np.random.seed(500 * (self.times + 1) + epoch + 1)
            torch.manual_seed(500 * (self.times + 1) + epoch + 1)
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

            self.optimizer.zero_grad()

            for p, grad in zip(self.model.parameters(), grads):
                grad += p.grad.data

        for control, grad in zip(self.controls, grads):
            control.data = grad / self.local_updates
        self.optimizer.zero_grad()

    def train_no_dp(self, glob_iter, user_ratio, warm_start, seen):
        """Training phase without differential privacy"""
        # no training during warm start strategy
        if (not warm_start) or glob_iter >= round(4 / user_ratio):
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

                self.optimizer.step(self.server_controls, self.controls)
                if self.scheduler:
                    self.scheduler.step()

        # get model difference
        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()

        # get user new controls
        new_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        for server_control, control, new_control, delta in zip(self.server_controls, self.controls, new_controls,
                                                               self.delta_model):
            a = self.sample_ratio / (self.local_updates * self.learning_rate)
            new_control.data = control.data - server_control.data - delta.data * a

        # get controls differences
        for control, new_control, delta in zip(self.controls, new_controls, self.delta_controls):
            if (not warm_start) or glob_iter >= round(4 / user_ratio):
                delta.data = new_control.data - control.data
            else:
                if not seen:
                    delta.data = new_control.data
                else:
                    delta.data = new_control.data - control.data
            control.data = new_control.data

        return 0

    def train_dp(self, sigma_g, glob_iter, user_ratio, max_norm, warm_start, seen):
        """Training phase under differential privacy"""
        # no training during warm start strategy
        if (not warm_start) or glob_iter >= round(4 / user_ratio):
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

                self.optimizer.step(self.server_controls, self.controls)

                if self.scheduler:
                    self.scheduler.step()

        # get model difference
        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()

        # get user new controls
        new_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        for server_control, control, new_control, delta in zip(self.server_controls, self.controls, new_controls,
                                                               self.delta_model):
            a = 1 / (self.local_updates * self.learning_rate)
            new_control.data = control.data - server_control.data - delta.data * a

        # get controls differences
        for control, new_control, delta in zip(self.controls, new_controls, self.delta_controls):
            if (not warm_start) or glob_iter >= round(4 / user_ratio):
                delta.data = new_control.data - control.data
            else:
                if not seen:
                    delta.data = new_control.data
                else:
                    delta.data = new_control.data - control.data
            control.data = new_control.data

        return 0

    def get_params_norm(self):
        """Returns (||x_user^t+1 -x_server^t||,||c_user^t+1 -c_server^t||)."""
        params = []
        controls = []

        for delta in self.delta_model:
            params.append(torch.flatten(delta.data))

        for delta in self.delta_controls:
            controls.append(torch.flatten(delta.data))

        return float(torch.norm(torch.cat(params))), float(torch.norm(torch.cat(controls)))
