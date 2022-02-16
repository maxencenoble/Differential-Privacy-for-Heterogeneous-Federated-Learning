import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np
import copy


# Super class for the user settings (either FedAvg/FedSGD or SCAFFOLD)

class User:

    def __init__(self, user_id, train_data, test_data, model, sample_ratio, learning_rate, L, local_updates,
                 dp, times, use_cuda):
        self.use_cuda = use_cuda

        self.optimizer = None
        self.model = copy.deepcopy(model)
        if use_cuda:
            self.model = self.model.cuda()
        self.user_id = user_id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.sample_ratio = sample_ratio
        self.batch_size = round(sample_ratio * self.train_samples)
        self.learning_rate = learning_rate
        self.L = L
        self.local_updates = local_updates
        self.scheduler = None
        self.lr_drop_rate = 1
        self.train_data = train_data
        self.times = times

        # for data loader
        np.random.seed(0)
        torch.manual_seed(0)
        train_idx = np.arange(self.train_samples)
        train_sampler = SubsetRandomSampler(train_idx)
        self.trainloader = DataLoader(train_data, self.batch_size, sampler=train_sampler)

        self.testloader = DataLoader(test_data, self.batch_size)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        self.dp = dp

        self.delta_model = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.server_model = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]

        # those parameters are for FEDL.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.server_grad = copy.deepcopy(list(self.model.parameters()))

    def set_parameters(self, server_model):
        for old_param, new_param, local_param, server_param in zip(self.model.parameters(), server_model.parameters(),
                                                                   self.local_model, self.server_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
            server_param.data = new_param.data.clone()
            if (new_param.grad != None):
                if (old_param.grad == None):
                    old_param.grad = torch.zeros_like(new_param.grad)

                if (local_param.grad == None):
                    local_param.grad = torch.zeros_like(new_param.grad)

                old_param.grad.data = new_param.grad.data.clone()
                local_param.grad.data = new_param.grad.data.clone()
        # self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def set_new_parameters(self, new_parameters):
        for old_param, new_param in zip(self.model.parameters(), new_parameters):
            old_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
            if (param.grad != None):
                if (clone_param.grad == None):
                    clone_param.grad = torch.zeros_like(param.grad)
                clone_param.grad.data = param.grad.data.clone()

        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()
            param.grad.data = new_param.grad.data.clone()

    def get_grads(self, grads):
        self.optimizer.zero_grad()

        for x, y in self.trainloaderfull:
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
        self.clone_model_paramenter(self.model.parameters(), grads)
        # for param, grad in zip(self.model.parameters(), grads):
        #    if(grad.grad == None):
        #        grad.grad = torch.zeros_like(param.grad)
        #    grad.grad.data = param.grad.data.clone()
        return grads

    def test_error_and_loss(self):
        """Returns metrics evaluated on test data."""
        self.model.eval()
        test_acc = 0
        loss = 0
        for x, y in self.testloaderfull:
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.user_id + ", Test Loss:", loss)
        return test_acc, loss, y.shape[0]

    def train_error_and_loss(self, model_lowest):
        """Returns metrics evaluated on train data."""
        self.model.eval()
        model_lowest.eval()
        train_acc = 0
        loss = 0
        loss_lowest = 0
        for x, y in self.trainloaderfull:
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()
            output_lowest = model_lowest(x)
            loss_lowest += self.loss(output_lowest, y)

            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.user_id + ", Train Accuracy:", train_acc)
            # print(self.user_id + ", Train Loss:", loss)
        return train_acc, loss, loss_lowest, self.train_samples

    def train_dissimilarity(self):
        """Returns gradients for gradient dissimilarity."""
        self.model.eval()
        gradients = [torch.flatten(torch.zeros_like(p.data)) for p in self.model.parameters()]
        for x, y in self.trainloaderfull:
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            for p, gradient in zip(self.model.parameters(), gradients):
                gradient += torch.flatten(copy.deepcopy(p.grad.data))

        return torch.cat(gradients)

    def get_next_train_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X, y)

    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def drop_lr(self):
        for group in self.optimizer.param_groups:
            group['lr'] *= self.lr_drop_rate
            if self.scheduler:
                group['initial_lr'] *= self.lr_drop_rate
