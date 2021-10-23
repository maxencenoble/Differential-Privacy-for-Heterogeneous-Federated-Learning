import os
import torch
import torch.nn as nn
import numpy as np
import copy
import random
from torch.utils.data import DataLoader
from torch.optim import LBFGS
from torch.optim import SGD
from utils.model_utils import read_data, read_user_data


class Optim:
    def __init__(self, dataset, model, number, similarity, alpha, beta):

        if similarity is None:
            similarity = (alpha, beta)
        if alpha < 0 and beta < 0:
            similarity = "iid"

        self.similarity = similarity
        self.dataset = dataset
        self.number = str(number)
        self.model = copy.deepcopy(model[0])
        self.learning_rate = 0.5

        if model[1] == 'mclr':
            self.loss = nn.NLLLoss()
            self.optimizer = LBFGS(self.model.parameters(), lr=self.learning_rate)
        else:
            self.loss = nn.CrossEntropyLoss()
            self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate)

        # Initialize data
        data = read_data(dataset, self.number, str(self.similarity))
        total_users = len(data[0])
        self.train_dataset = []
        self.test_dataset = []

        for i in range(total_users):
            id, train, test = read_user_data(i, data, dataset)
            self.train_dataset += train
            self.test_dataset += test
        
        random.shuffle(self.train_dataset)
        random.shuffle(self.test_dataset)

        self.batch_size = 500
        self.epochs = 2000
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=False)
        self.train_loader_full = DataLoader(dataset=self.train_dataset, batch_size=len(self.train_dataset))
        self.test_loader_full = DataLoader(dataset=self.test_dataset, batch_size=len(self.test_dataset))

    def train(self):
        lowest_loss = 1e3
        for epoch in range(int(self.epochs)):
            # calculate Accuracy
            correct = 0
            total = 0
            loss = 0

            # train
            for i, (images, labels) in enumerate(self.train_loader):
                def closure():
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.loss(outputs, labels)
                    loss.backward()
                    return loss

                self.optimizer.step(closure)

            # validation
            for i, (images, labels) in enumerate(self.train_loader_full):
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print("TRAIN : Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

            if loss.item() < lowest_loss or epoch == 0:
                self.save_model()
                lowest_loss = copy.copy(loss.item())

            # test
            correct = 0
            total = 0
            loss = 0
            for i, (images, labels) in enumerate(self.test_loader_full):
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print("TEST : Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server_lowest_" + str(self.similarity) + ".pt"))
