import torch
import torch.nn as nn
import torch.nn.functional as F


class MclrLogistic(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(MclrLogistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


# one hidden layer

class NN1(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(NN1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class NN1_PCA(nn.Module):
    def __init__(self, input_dim=60, output_dim=10):
        super(NN1_PCA, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# CNN

class CNN(nn.Module):
    def __init__(self, output_dim=10, inter_dim=200):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride=1)
        self.conv2 = nn.Conv2d(64, 64, 5, stride=1)
        self.fc1 = nn.Linear(4 * 4 * 64, inter_dim)
        self.fc2 = nn.Linear(inter_dim, output_dim)

    def forward(self, x):
        x = torch.reshape(x, (-1, 3, 28, 28))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
