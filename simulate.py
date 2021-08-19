#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from flearn.servers.optimum import Optim
from flearn.servers.server_avg import FedAvg
from flearn.servers.server_scaffold import SCAFFOLD
from flearn.trainmodel.models import *
from utils.plot_utils import *
from utils.autograd_hacks import *
import torch

torch.manual_seed(0)  # for initialisation of the models


def find_optimum(dataset, model, number, dim_input, dim_output, similarity=None, alpha=0., beta=0.):
    # Generate model
    if model == "mclr":  # for Femnist dataset, Logistic datasets
        model = MclrLogistic(input_dim=dim_input, output_dim=dim_output), model

    if model == "dnn":  # for Femnist dataset
        model = DNN(input_dim=dim_input, output_dim=dim_output), model

    if model == "net":  # for Femnist dataset
        model = Net(output_dim=dim_output), model

    if model == "net2":  # for Femnist dataset
        model = Net2(output_dim=dim_output), model

    if model == "resnet":  # for Femnist dataset
        model = MyResNet18(output_dim=dim_output), model

    if model == "cnn":  # for Femnist dataset
        model = CNN(output_dim=dim_output), model

    server = Optim(dataset, model, number, similarity, alpha, beta)
    server.train()


def simulate(dataset, algorithm, model, dim_input, dim_output, nb_users, nb_samples, sample_ratio, user_ratio,
             weight_decay, local_learning_rate, max_norm, local_updates, noise, times, dp, epsilon_target,
             similarity=None, alpha=0.,beta=0., number=0, num_glob_iters=400):
    users_per_round = int(nb_users * user_ratio)
    L = weight_decay

    print("=" * 80)
    print("Summary of training process:")
    print(f"Algorithm: {algorithm}")
    print(f"Subset of users         : {users_per_round if users_per_round else 'all users'}")
    print(f"Number of local rounds  : {local_updates}")
    print(f"Number of global rounds : {num_glob_iters}")
    print(f"Dataset                 : {dataset}")
    if similarity is not None:
        print(f"Data Similarity         : {similarity}")
    else:
        print(f"Data Similarity         : {(alpha, beta)}")
    print(f"Local Model             : {model}")
    print("=" * 80)

    for i in range(times):
        print("---------------Running time:------------", i)

        # Generate model
        if model == "mclr":  # for Femnist dataset, Logistic dataset
            model = MclrLogistic(input_dim=dim_input, output_dim=dim_output), model
            add_hooks(model[0])

        if model == "dnn":  # for Femnist dataset
            model = DNN(input_dim=dim_input, output_dim=dim_output), model
            add_hooks(model[0])

        if model == "net":  # for Femnist dataset
            model = Net(output_dim=dim_output), model
            add_hooks(model[0])

        if model == "net2":  # for Femnist dataset
            model = Net2(output_dim=dim_output), model
            add_hooks(model[0])

        if model == "resnet":  # for Femnist dataset
            model = MyResNet18(output_dim=dim_output), model
            add_hooks(model[0])

        if model == "cnn":  # for Femnist dataset
            model = CNN(output_dim=dim_output), model
            add_hooks(model[0])

        # select algorithm
        if algorithm == "FedAvg":
            server = FedAvg(dataset, algorithm, model, nb_users, nb_samples, user_ratio, sample_ratio, L,
                            local_learning_rate, max_norm,
                            num_glob_iters, local_updates, users_per_round, similarity, noise, i, dp,
                            epsilon_target, alpha, beta, number)

        elif algorithm == "SCAFFOLD":
            server = SCAFFOLD(dataset, algorithm, model, nb_users, nb_samples, user_ratio, sample_ratio, L,
                              local_learning_rate, max_norm,
                              num_glob_iters, local_updates, users_per_round, similarity, noise, i, dp,
                              epsilon_target, alpha, beta, number, warm_start=False)
        elif algorithm == "SCAFFOLD-warm":
            server = SCAFFOLD(dataset, algorithm, model, nb_users, nb_samples, user_ratio, sample_ratio, L,
                              local_learning_rate, max_norm,
                              num_glob_iters, local_updates, users_per_round, similarity, noise, i, dp,
                              epsilon_target, alpha, beta, number, warm_start=True)
        server.train()

    # Average data

    if similarity is None:
        similarity = (alpha, beta)

    if alpha < 0. and beta < 0.:
        similarity = "iid"

    average_data(num_glob_iters=num_glob_iters, algorithm=algorithm, dataset=dataset, similarity=similarity,
                 noise=noise, times=times, number=str(number), dp=dp, epsilon=epsilon_target, local_updates=local_updates,
                 sample_ratio=sample_ratio)
    average_norms(num_glob_iters=num_glob_iters, algorithm=algorithm, dataset=dataset, similarity=similarity,
                  noise=noise, times=times, number=str(number), dp=dp, epsilon=epsilon_target,
                  local_updates=local_updates, sample_ratio=sample_ratio)