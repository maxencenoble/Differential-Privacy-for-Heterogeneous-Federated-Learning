#!/usr/bin/env python
from flearn.servers.optimum import Optim
from flearn.servers.server_avg import FedAvg
from flearn.servers.server_scaffold import SCAFFOLD
from flearn.trainmodel.models import *
from utils.plot_utils import *
from utils.autograd_hacks import *
import torch


def find_optimum(dataset, model, number, dim_input, dim_output, dim_pca=None, similarity=None, alpha=0., beta=0.):
    torch.manual_seed(0)  # for initialisation of the models
    use_cuda = torch.cuda.is_available()

    # Generate model
    if model == "mclr":  # for Femnist, MNIST, Logistic datasets
        model = MclrLogistic(input_dim=dim_input, output_dim=dim_output), model

    if model == "NN1":  # for Femnist, MNIST datasets
        model = NN1(input_dim=dim_input, output_dim=dim_output), model

    if model == "NN1_PCA":  # for Femnist, MNIST dataset
        model = NN1_PCA(input_dim=dim_pca, output_dim=dim_output), model

    if model == "CNN":  # for CIFAR-10 dataset
        model = CNN(output_dim=dim_output), model

    if use_cuda:
        print("Using GPU")
    server = Optim(dataset, model, number, similarity, alpha, beta, dim_pca, use_cuda)
    server.train()


def simulate(dataset, algorithm, model, dim_input, dim_output, nb_users, nb_samples, sample_ratio, user_ratio,
             weight_decay, local_learning_rate, max_norm, local_updates, noise, times, dp, sigma_gaussian, dim_pca,
             similarity=None, alpha=0., beta=0., number=0, num_glob_iters=400, time=None):
    users_per_round = int(nb_users * user_ratio)
    L = weight_decay

    print("=" * 80)
    print("Summary of training process:")
    print(f"Algorithm: {algorithm}")
    if dp == "Gaussian":
        print(f"Differential Privacy : Gaussian Mechanism with sigma_g={sigma_gaussian}")
    else:
        print(f"Noise free version")
    print(f"Subset of users                : {users_per_round if users_per_round else 'all users'}")
    print(f"Number of local updates        : {local_updates}")
    print(f"Number of communication rounds : {num_glob_iters}")
    print(f"Dataset                        : {dataset}")
    if similarity is not None:
        print(f"Data Similarity                : {similarity}")
    else:
        print(f"Data Similarity                : {(alpha, beta)}")
    print(f"Local Model                    : {model}")
    print("=" * 80)

    beg = 0
    end = times

    # to process only 1 run
    if time is not None:
        beg = time
        end = min(time + 1, times)

    for i in range(beg, end):
        torch.manual_seed(0)  # for initialisation of the models
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print("Using GPU")
        print("---------------Running time:------------", i)

        # Generate model
        # add_hooks: useful to get per-sample gradients

        if model == "mclr":  # for Femnist, MNIST, Logistic datasets
            model = MclrLogistic(input_dim=dim_input, output_dim=dim_output), model
            add_hooks(model[0])

        if model == "NN1":  # for Femnist, MNIST datasets
            model = NN1(input_dim=dim_input, output_dim=dim_output), model
            add_hooks(model[0])

        if model == "NN1_PCA":  # for Femnist, MNIST datasets
            model = NN1_PCA(input_dim=dim_pca, output_dim=dim_output), model
            add_hooks(model[0])

        if model == "CNN":  # for CIFAR-10 dataset
            model = CNN(output_dim=dim_output), model
            add_hooks(model[0])

        # select algorithm

        if algorithm == "FedAvg" or algorithm == "FedSGD":
            server = FedAvg(dataset, algorithm, model, nb_users, nb_samples, user_ratio, sample_ratio, L,
                            local_learning_rate, max_norm, num_glob_iters, local_updates, users_per_round, similarity,
                            noise, i, dp, sigma_gaussian, alpha, beta, number, dim_pca, use_cuda)

        elif algorithm == "SCAFFOLD":
            server = SCAFFOLD(dataset, algorithm, model, nb_users, nb_samples, user_ratio, sample_ratio, L,
                              local_learning_rate, max_norm, num_glob_iters, local_updates, users_per_round, similarity,
                              noise, i, dp, sigma_gaussian, alpha, beta, number, dim_pca, use_cuda, warm_start=False)

        elif algorithm == "SCAFFOLD-warm":
            server = SCAFFOLD(dataset, algorithm, model, nb_users, nb_samples, user_ratio, sample_ratio, L,
                              local_learning_rate, max_norm, num_glob_iters, local_updates, users_per_round, similarity,
                              noise, i, dp, sigma_gaussian, alpha, beta, number, dim_pca, use_cuda, warm_start=True)
        server.train()

    # Average results

    if similarity is None:
        similarity = (alpha, beta)

    if alpha < 0. and beta < 0.:
        similarity = "iid"

    average_data(num_glob_iters=num_glob_iters, algorithm=algorithm, dataset=dataset, similarity=similarity,
                 noise=noise, times=times, number=str(number), dp=dp, sigma_gaussian=sigma_gaussian,
                 local_updates=local_updates, sample_ratio=sample_ratio, user_ratio=user_ratio, model_name=model[1])
    average_norms(num_glob_iters=num_glob_iters, algorithm=algorithm, dataset=dataset, similarity=similarity,
                  noise=noise, times=times, number=str(number), dp=dp, sigma_gaussian=sigma_gaussian,
                  local_updates=local_updates, sample_ratio=sample_ratio,user_ratio=user_ratio, model_name=model[1])


def simulate_cross_validation(dataset, algorithm, model, dim_input, dim_pca, dim_output, nb_users, nb_samples,
                              sample_ratio, user_ratio, weight_decay, local_learning_rate, max_norm, local_updates,
                              noise, times, dp, sigma_gaussian, similarity=None, alpha=0., beta=0., number=0,
                              num_glob_iters=400, nb_fold=5):
    users_per_round = int(nb_users * user_ratio)
    L = weight_decay

    print("=" * 80)
    print("Summary of training process:")
    print(f"Algorithm: {algorithm}")
    if dp == "Gaussian":
        print(f"Differential Privacy : Gaussian Mechanism with sigma_g={sigma_gaussian}")
    else:
        print(f"Noise free version")
    print(f"Subset of users                : {users_per_round if users_per_round else 'all users'}")
    print(f"Number of local updates        : {local_updates}")
    print(f"Number of communication rounds : {num_glob_iters}")
    print(f"Dataset                        : {dataset}")
    if similarity is not None:
        print(f"Data Similarity                : {similarity}")
    else:
        print(f"Data Similarity                : {(alpha, beta)}")
    print(f"Local Model                    : {model}")
    print("=" * 80)

    for k_fold in np.arange(nb_fold):
        print("----------CROSS VALIDATION: {}/{} ".format(k_fold + 1, nb_fold))
        for i in range(times):
            torch.manual_seed(0)  # for initialisation of the models
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                print("Using GPU")
            print("---------------Running time:------------", i)

            # Generate model
            if model == "mclr":  # for Femnist, MNIST, Logistic datasets
                model = MclrLogistic(input_dim=dim_input, output_dim=dim_output), model
                add_hooks(model[0])

            if model == "NN1":  # for Femnist, MNIST datasets
                model = NN1(input_dim=dim_input, output_dim=dim_output), model
                add_hooks(model[0])

            if model == "NN1_PCA":  # for Femnist, MNIST datasets
                model = NN1_PCA(input_dim=dim_pca, output_dim=dim_output), model
                add_hooks(model[0])

            if model == "CNN":  # for CIFAR-10 dataset
                model = CNN(output_dim=dim_output), model
                add_hooks(model[0])

            # select algorithm
            if algorithm == "FedAvg":
                server = FedAvg(dataset, algorithm, model, nb_users, nb_samples, user_ratio, sample_ratio, L,
                                local_learning_rate, max_norm, num_glob_iters, local_updates, users_per_round,
                                similarity, noise, i, dp, sigma_gaussian, alpha, beta, number, dim_pca, use_cuda,
                                k_fold=k_fold, nb_fold=nb_fold)

            elif algorithm == "SCAFFOLD":
                server = SCAFFOLD(dataset, algorithm, model, nb_users, nb_samples, user_ratio, sample_ratio, L,
                                  local_learning_rate, max_norm, num_glob_iters, local_updates, users_per_round,
                                  similarity, noise, i, dp, sigma_gaussian, alpha, beta, number, dim_pca, use_cuda,
                                  warm_start=False,
                                  k_fold=k_fold, nb_fold=nb_fold)
            server.train()

        # Average results

        if similarity is None:
            similarity = (alpha, beta)

        if alpha < 0. and beta < 0.:
            similarity = "iid"

        average_data(num_glob_iters=num_glob_iters, algorithm=algorithm, dataset=dataset,
                     similarity=similarity, noise=noise, times=times, number=str(number), dp=dp,
                     sigma_gaussian=sigma_gaussian, local_updates=local_updates, sample_ratio=sample_ratio,
                     user_ratio=user_ratio,
                     cross_validation=True, k_fold=k_fold, nb_fold=nb_fold, local_learning_rate=local_learning_rate,
                     model_name=model[1])
