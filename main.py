from utils.plot_utils import *
import argparse
from simulate import simulate
from simulate import simulate_cross_validation
from simulate import find_optimum
from data.Mnist.data_generator import generate_data as generate_mnist_data
from data.Mnist.data_generator import generate_pca_data as generate_mnist_pca_data
from data.Femnist.data_generator import generate_data as generate_femnist_data
from data.Femnist.data_generator import generate_pca_data as generate_femnist_pca_data
from data.CIFAR_10.data_generator import generate_data as generate_cifar10_data
from data.Logistic.data_generator import generate_data as generate_logistic_data


def generate_data(dataset, nb_users, nb_samples, dim_input=40, dim_output=10, similarity=1.0, alpha=0., beta=0.,
                  number=0, iid=False, same_sample_size=True, normalise=False, standardize=False):
    if dataset == 'Femnist':
        generate_femnist_data(similarity, num_users=nb_users, num_samples=nb_samples, number=number,
                              normalise=normalise)
    elif dataset == 'Mnist':
        generate_mnist_data(similarity, num_users=nb_users, num_samples=nb_samples, number=number,
                            normalise=normalise)
    elif dataset == 'CIFAR_10':
        generate_cifar10_data(similarity, num_users=nb_users, num_samples=nb_samples, number=number)

    elif dataset == 'Logistic':
        generate_logistic_data(num_users=nb_users, same_sample_size=same_sample_size, num_samples=nb_samples,
                               dim_input=dim_input, dim_output=dim_output, alpha=alpha, beta=beta, number=number,
                               normalise=normalise, standardize=standardize, iid=iid)


def run_simulation(time, dataset, algo, model, similarity, alpha, beta, number, dim_input, dim_output, same_sample_size,
                   nb_users, user_ratio, nb_samples, sample_ratio, local_updates, weight_decay, local_learning_rate,
                   max_norm, dp, sigma_gaussian, normalise, standardize, times, optimum, num_glob_iters, generate,
                   generate_pca, dim_pca, tuning, learning, plot):
    if dataset == "Femnist":
        nb_users = 40
        nb_samples = 2500

    if dataset == "Mnist":
        nb_users = 60
        nb_samples = 1000

    if dataset == "CIFAR_10":
        nb_users = 50
        nb_samples = 1000

    # FEMNIST DATA
    # Potential models : mclr, NN1, NN1_PCA

    femnist_dict = {"dataset": "Femnist",
                    "model": model,
                    "dim_input": 784,
                    "dim_pca": dim_pca,
                    "dim_output": 47,
                    "nb_users": nb_users,
                    "nb_samples": nb_samples,
                    "sample_ratio": sample_ratio,
                    "local_updates": local_updates,
                    "user_ratio": user_ratio,
                    "weight_decay": weight_decay,
                    "local_learning_rate": local_learning_rate,
                    "max_norm": max_norm}

    # MNIST DATA
    # Potential models : mclr, NN1, NN1_PCA

    mnist_dict = {"dataset": "Mnist",
                  "model": model,
                  "dim_input": 784,
                  "dim_pca": dim_pca,
                  "dim_output": 10,
                  "nb_users": nb_users,
                  "nb_samples": nb_samples,
                  "sample_ratio": sample_ratio,
                  "local_updates": local_updates,
                  "user_ratio": user_ratio,
                  "weight_decay": weight_decay,
                  "local_learning_rate": local_learning_rate,
                  "max_norm": max_norm}

    # CIFAR-10 DATA
    # Potential models : CNN

    cifar10_dict = {"dataset": "CIFAR_10",
                    "model": "CNN",
                    "dim_input": 1024,
                    "dim_pca": None,
                    "dim_output": 10,
                    "nb_users": nb_users,
                    "nb_samples": nb_samples,
                    "sample_ratio": sample_ratio,
                    "local_updates": local_updates,
                    "user_ratio": user_ratio,
                    "weight_decay": weight_decay,
                    "local_learning_rate": local_learning_rate,
                    "max_norm": max_norm}

    # SYNTHETIC DATA
    # only one model : mclr

    logistic_dict = {"dataset": "Logistic",
                     "model": "mclr",
                     "dim_input": dim_input,
                     "dim_pca": None,
                     "dim_output": dim_output,
                     "nb_users": nb_users,
                     "nb_samples": nb_samples,
                     "sample_ratio": sample_ratio,
                     "local_updates": local_updates,
                     "user_ratio": user_ratio,
                     "weight_decay": weight_decay,
                     "local_learning_rate": local_learning_rate,
                     "max_norm": max_norm}

    input_dict = {}

    if dataset == 'Femnist':
        input_dict = femnist_dict
    elif dataset == 'Mnist':
        input_dict = mnist_dict
    elif dataset == 'Logistic':
        input_dict = logistic_dict
    elif dataset == 'CIFAR_10':
        input_dict = cifar10_dict

    algorithms = ["FedAvg", "SCAFFOLD"]
    dps = [dp]
    noises = [False]
    similarities = [1.0, 0.1, 0.0] # for Femnist, Mnist, CIFAR data
    if dp == "None":
        local_learning_rate_list = []  # TO FILL for tuning (without dp)
    elif dp == "Gaussian":
        local_learning_rate_list = []  # TO FILL for tuning (with dp)
    # for logistic data
    alphas = [0.0, 1.0, 5.0]  # heterogeneity : between models
    betas = [0.0, 1.0, 5.0]  # heterogeneity : between data records
    # to set iid synthetic data, choose alpha=-1.0 and beta=-1.0

    if dataset in ['Logistic']:
        similarities = list(zip(alphas, betas))

    if generate:
        if dataset in ['Femnist', 'Mnist', 'CIFAR_10']:
            for similarity in similarities:
                generate_data(dataset=dataset, nb_users=nb_users, nb_samples=nb_samples, similarity=similarity,
                              number=number, normalise=normalise)

        if dataset in ['Logistic']:
            similarities = list(zip(alphas, betas))
            for similarity in similarities:
                alpha, beta = similarity
                iid = False
                if alpha < 0 and beta < 0:
                    iid = True
                generate_data(dataset=dataset, nb_users=nb_users, nb_samples=nb_samples, dim_input=dim_input,
                              alpha=alpha, beta=beta, number=number, same_sample_size=same_sample_size,
                              normalise=normalise, standardize=standardize, dim_output=dim_output, iid=iid)

    elif generate_pca and dataset in ['Femnist']:
        for similarity in similarities:
            generate_femnist_pca_data(similarity, dim_pca=dim_pca, num_users=nb_users, num_samples=nb_samples,
                                      number=number, normalise=normalise)

    elif generate_pca and dataset in ['Mnist']:
        for similarity in similarities:
            generate_mnist_pca_data(similarity, dim_pca=dim_pca, num_users=nb_users, num_samples=nb_samples,
                                    number=number, normalise=normalise)

    elif optimum:
        if dataset in ['Femnist', 'Mnist', 'CIFAR_10']:
            for similarity in similarities:
                find_optimum(dataset=dataset, model=input_dict["model"], number=number,
                             dim_input=input_dict["dim_input"],
                             dim_output=input_dict["dim_output"], similarity=similarity, dim_pca=dim_pca)

        if dataset in ['Logistic']:
            for similarity in similarities:
                alpha, beta = similarity
                find_optimum(dataset=dataset, model=logistic_dict["model"], number=number,
                             dim_input=logistic_dict["dim_input"],
                             dim_output=logistic_dict["dim_output"], alpha=alpha, beta=beta)

    elif tuning:
        if dataset in ['Femnist', 'Mnist', 'CIFAR_10']:
            for similarity in similarities:
                for noise in noises:
                    for dp in dps:
                        for algorithm in algorithms:
                            for lr in local_learning_rate_list:
                                input_dict["local_learning_rate"] = lr
                                print("Hyperparameter :{}".format(lr))
                                simulate_cross_validation(**input_dict, algorithm=algorithm, similarity=similarity,
                                                          noise=noise, times=times, dp=dp,
                                                          sigma_gaussian=sigma_gaussian,
                                                          num_glob_iters=num_glob_iters)

        if dataset in ['Logistic']:
            for similarity in similarities:
                alpha, beta = similarity
                for noise in noises:
                    for dp in dps:
                        for algorithm in algorithms:
                            for lr in local_learning_rate_list:
                                input_dict["local_learning_rate"] = lr
                                print("Hyperparameter :{}".format(lr))
                                simulate_cross_validation(**input_dict, algorithm=algorithm, noise=noise,
                                                          times=times, dp=dp, sigma_gaussian=sigma_gaussian,
                                                          alpha=alpha, beta=beta,
                                                          similarity=None, number=number, num_glob_iters=num_glob_iters)

    elif learning:
        if dataset in ['Femnist', 'Mnist', 'CIFAR_10']:
            for similarity in similarities:
                for noise in noises:
                    for dp in dps:
                        for algorithm in algorithms:
                            if algorithm == "FedSGD":  # SGD: one local epoch
                                input_dict["local_updates"] = round(1 / input_dict["sample_ratio"])
                            simulate(**input_dict, algorithm=algorithm, similarity=similarity, noise=noise,
                                     times=times, dp=dp, sigma_gaussian=sigma_gaussian,
                                     num_glob_iters=num_glob_iters, time=time)

        if dataset in ['Logistic']:
            for similarity in similarities:
                alpha, beta = similarity
                for noise in noises:
                    for dp in dps:
                        for algorithm in algorithms:
                            if algorithm == "FedSGD":  # SGD: one local epoch
                                input_dict["local_updates"] = round(1 / input_dict["sample_ratio"])
                            simulate(**input_dict, algorithm=algorithm, noise=noise,
                                     times=times, dp=dp, sigma_gaussian=sigma_gaussian, alpha=alpha, beta=beta,
                                     similarity=None, number=number, num_glob_iters=num_glob_iters, time=time)
    elif plot:

        # Plots with same sigma_gaussian, same T, same K, same l, same s + various similarities
        plot_test_accuracy(dataset, algorithms, noises, similarities, str(number), sigma_gaussian, local_updates,
                           sample_ratio, user_ratio, input_dict["model"])
        plot_train_loss(dataset, algorithms, noises, similarities, str(number), sigma_gaussian, local_updates,
                        sample_ratio, user_ratio, input_dict["model"])
        plot_norms(dataset, algorithms, noises, similarities, str(number), sigma_gaussian, local_updates,
                   sample_ratio, user_ratio, input_dict["model"])
        plot_train_dissimilarity(dataset, algorithms, noises, similarities, str(number), sigma_gaussian,
                                 local_updates, sample_ratio, user_ratio, input_dict["model"])

        # Plots with same sigma_gaussian, same K, same l + various similarities, various s, various T

        list_of_sample_ratio = []  # TO FILL, for instance: 0.05, 0.1, 0.2
        if len(list_of_sample_ratio) > 0:
            plot_test_accuracy_multiple_sample_ratio(dataset, algorithms, noises, similarities, str(number),
                                                     sigma_gaussian,
                                                     local_updates, list_of_sample_ratio, user_ratio,
                                                     input_dict["model"])
            plot_train_loss_multiple_sample_ratio(dataset, algorithms, noises, similarities, str(number),
                                                  sigma_gaussian, local_updates, list_of_sample_ratio, user_ratio,
                                                  input_dict["model"])

        # Plots with same sigma_gaussian, same K, same s + various similarities, various l, various T

        list_of_user_ratio = []  # TO FILL: for instance, 0.12, 0.1, 0.08
        if len(list_of_user_ratio) > 0:
            plot_test_accuracy_multiple_user_ratio(dataset, algorithms, noises, similarities, str(number),
                                                   sigma_gaussian,
                                                   local_updates, sample_ratio, list_of_user_ratio,
                                                   input_dict["model"])
            plot_train_loss_multiple_user_ratio(dataset, algorithms, noises, similarities, str(number),
                                                sigma_gaussian, local_updates, sample_ratio, list_of_user_ratio,
                                                input_dict["model"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--times", type=int, default=3, help="Number of random runs for each setting")
    parser.add_argument("--time", type=int, default=None, choices=[None, 0, 1, 2],
                        help="(<times) : used to process the run chosen independently from the others. If None, every run is performed")
    parser.add_argument("--num_glob_iters", type=int, default=250, help="T: Number of communication rounds")

    parser.add_argument("--dataset", type=str, default="Logistic", choices=["Femnist", "Logistic", "CIFAR_10", "Mnist"])
    parser.add_argument("--algo", type=str, default="FedAvg", choices=["FedSGD", "FedAvg", "SCAFFOLD-warm", "SCAFFOLD"])
    parser.add_argument("--model", type=str, default="mclr", choices=["mclr", "NN1", "NN1_PCA", "CNN"],
                        help="Chosen model. If using PCA on data, add '_PCA' at the end of the name.")
    parser.add_argument("--similarity", type=float, default=0.1,
                        help="Level of similarity between user data (for Femnist, Mnist, CIFAR_10 datasets)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Level of heterogeneity between user model (for Logistic dataset), -1 for iid models")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Level of heterogeneity between user data (for Logistic dataset), -1 for iid data")
    parser.add_argument("--number", type=int, default=0,
                        help="Id of dataset (used to avoid overwriting if same similarity parameters are used)")

    parser.add_argument("--nb_users", type=int, default=100, help="M: Number of all users for FL")
    # In the paper: FEMNIST : 40 users / Logistic : 100 users
    parser.add_argument("--user_ratio", type=float, default=0.1,
                        help="l: Subsampling ratio for users at each communication round")
    parser.add_argument("--nb_samples", type=int, default=5000,
                        help="R: Number of all data points by user (conditionally to same_sample_size)")
    # In the paper: FEMNIST : 2500 samples / Logistic : 5000 samples
    parser.add_argument("--sample_ratio", type=float, default=0.2,
                        help="s: Subsampling ratio for data points at each local update")
    parser.add_argument("--local_updates", type=int, default=10,
                        help="K: Number of local updates per selected user (local_epochs=local_updates*sample_ratio)")

    # For Logistic dataset generation
    parser.add_argument("--dim_input", type=int, default=40, help="For synthetic data : size of data points")
    parser.add_argument("--dim_output", type=int, default=10, help="For synthetic data : nb of labels")
    parser.add_argument("--same_sample_size", type=int, default=1,
                        help="For synthetic data (generation): same sample size for all users?")
    # For both datasets generation
    parser.add_argument("--normalise", type=int, default=1,
                        help="If 1: Normalise every input at the generation of the data")
    parser.add_argument("--standardize", type=int, default=1,
                        help="If 1: Standardize the features by user at the generation of the data")

    parser.add_argument("--weight_decay", type=float, default=5e-3, help="Regularization term")
    parser.add_argument("--local_learning_rate", type=float, default=1.0,
                        help="Multiplicative factor in the learning rate for local updates (TO TUNE)")
    parser.add_argument("--max_norm", type=float, default=1.0,
                        help="Gradient clipping value (not used with the heuristic implemented by default)")

    parser.add_argument("--dp", type=str, default="None", choices=["None", "Gaussian"],
                        help="Differential Privacy or not")
    parser.add_argument("--sigma_gaussian", type=float, default=10.0, help="Gaussian standard deviation for DP noise")

    parser.add_argument("--generate", type=int, default=0,
                        help="True to generate data")
    parser.add_argument("--generate_pca", type=int, default=0,
                        help="True to generate data with PCA (for MNIST and FEMNIST data)")
    parser.add_argument("--dim_pca", type=int, default=60,
                        help="Nb of components for generate_pca (for MNIST and FEMNIST data)")
    parser.add_argument("--optimum", type=int, default=0,
                        help="True to train the model in a centralized setting and save the best model (data needed)")
    parser.add_argument("--tuning", type=int, default=0,
                        help="True to run tuning of hyperparameter 'local_learning_rate' (data needed)")
    parser.add_argument("--learning", type=int, default=0,
                        help="True to run learning (data needed, assuming tuning has been made)")
    parser.add_argument("--plot", type=int, default=0,
                        help="True to have plots (data needed, assuming learning has been made)")

    args = parser.parse_args()

    run_simulation(time=args.time, dataset=args.dataset, algo=args.algo, model=args.model, similarity=args.similarity,
                   alpha=args.alpha, beta=args.beta, number=args.number, dim_input=args.dim_input,
                   same_sample_size=args.same_sample_size, dim_output=args.dim_output,
                   nb_users=args.nb_users, user_ratio=args.user_ratio, nb_samples=args.nb_samples,
                   sample_ratio=args.sample_ratio, local_updates=args.local_updates, weight_decay=args.weight_decay,
                   local_learning_rate=args.local_learning_rate,
                   max_norm=args.max_norm, dp=args.dp, sigma_gaussian=args.sigma_gaussian, normalise=args.normalise,
                   times=args.times, standardize=args.standardize,
                   optimum=args.optimum, num_glob_iters=args.num_glob_iters,
                   generate=args.generate, tuning=args.tuning, learning=args.learning, plot=args.plot,
                   generate_pca=args.generate_pca, dim_pca=args.dim_pca)
