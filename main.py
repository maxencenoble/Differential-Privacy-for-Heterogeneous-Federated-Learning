from utils.plot_utils import *
import argparse
from simulate import simulate
from simulate import find_optimum
from data.Femnist.data_generator import generate_data as generate_femnist_data
from data.Logistic.data_generator import generate_data as generate_logistic_data


def generate_data(dataset, nb_users, nb_samples, dim_input=40, dim_output=10, similarity=1.0, alpha=0., beta=0.,
                  number=0, iid=False, same_sample_size=True, normalise=False, standardize=False):
    if dataset == 'Femnist':
        generate_femnist_data(similarity, num_users=nb_users, num_samples=nb_samples, number=number,
                              normalise=normalise)
    elif dataset == 'Logistic':
        generate_logistic_data(num_users=nb_users, same_sample_size=same_sample_size, num_samples=nb_samples,
                               dim_input=dim_input, dim_output=dim_output,
                               alpha=alpha, beta=beta, number=number, normalise=normalise, standardize=standardize,
                               iid=iid)


def run_simulation(dataset, number, dim_input, dim_output, same_sample_size, nb_users, user_ratio, nb_samples,
                   sample_ratio, local_updates, weight_decay, local_learning_rate, max_norm, dp, epsilon_target,
                   normalise, standardize, times, optimum, num_glob_iters, generate, simulation, plot):
    # Real world data
    # Potential models : mclr, resnet, net, net2, dnn , cnn
    # Best Convex : mclr
    # Best Non-Convex : resnet

    femnist_dict = {"dataset": "Femnist",
                    "model": "mclr",
                    "dim_input": 784,
                    "dim_output": 10,
                    "nb_users": nb_users,
                    "nb_samples": nb_samples,
                    "sample_ratio": sample_ratio,
                    "local_updates": local_updates,
                    "user_ratio": user_ratio,
                    "weight_decay": weight_decay,
                    "local_learning_rate": local_learning_rate,
                    "max_norm": max_norm}

    # Synthetic data
    # Only one model : mclr

    logistic_dict = {"dataset": "Logistic",
                     "model": "mclr",
                     "dim_input": dim_input,
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
    elif dataset == 'Logistic':
        input_dict = logistic_dict

    algorithms = ["SCAFFOLD", "SCAFFOLD-warm", "FedAvg"]
    dps = [dp]
    noises = [False]
    similarities = [0.5, 0.1, 0.0]
    alphas = [0.0, 1.0, 5.0]
    betas = [0.0, 0.0, 0.0]
    # to set iid synthetic data, choose alpha=-1.0 and beta=-1.0
    if dataset in ['Logistic']:
        similarities = list(zip(alphas, betas))

    if generate:
        if dataset in ['Femnist']:
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

    elif optimum:
        if dataset in ['Femnist']:
            for similarity in similarities:
                find_optimum(dataset=dataset, model=femnist_dict["model"], number=number,
                             dim_input=femnist_dict["dim_input"],
                             dim_output=femnist_dict["dim_output"], similarity=similarity)

        if dataset in ['Logistic']:
            for similarity in similarities:
                alpha, beta = similarity
                find_optimum(dataset=dataset, model=logistic_dict["model"], number=number,
                             dim_input=logistic_dict["dim_input"],
                             dim_output=logistic_dict["dim_output"], alpha=alpha, beta=beta)

    elif simulation:
        if dataset in ['Femnist']:
            for similarity in similarities:
                for noise in noises:
                    for dp in dps:
                        for algorithm in algorithms:
                            simulate(**input_dict, algorithm=algorithm, similarity=similarity, noise=noise,
                                     times=times, dp=dp, epsilon_target=epsilon_target,
                                     num_glob_iters=num_glob_iters)

        if dataset in ['Logistic']:
            for similarity in similarities:
                alpha, beta = similarity
                for noise in noises:
                    for dp in dps:
                        for algorithm in algorithms:
                            simulate(**input_dict, algorithm=algorithm, noise=noise,
                                     times=times, dp=dp, epsilon_target=epsilon_target, alpha=alpha, beta=beta,
                                     similarity=None, number=number, num_glob_iters=num_glob_iters)
    elif plot:
        plot_test_accuracy(dataset, algorithms, noises, similarities, str(number), epsilon_target, local_updates,
                           sample_ratio)
        plot_train_loss(dataset, algorithms, noises, similarities, str(number), epsilon_target, local_updates,
                        sample_ratio)
        plot_norms(dataset, algorithms, noises, similarities, str(number), epsilon_target, local_updates,
                   sample_ratio)
        plot_train_dissimilarity(dataset, algorithms, noises, similarities, str(number), epsilon_target,
                                 local_updates, sample_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Logistic",
                        choices=["Femnist", "Logistic"])
    parser.add_argument("--number", type=int, default=0,
                        help="Number of dataset (>0 if several datasets with same parameters)")

    parser.add_argument("--dim_input", type=int, default=40, help="For synthetic data : size of data points")
    parser.add_argument("--dim_output", type=int, default=10, help="For synthetic data : nb of labels")
    parser.add_argument("--same_sample_size", type=int, default=1,
                        help="For synthetic data (generation) : same sample size for all users")

    parser.add_argument("--normalise", type=int, default=1,
                        help="Normalise every input at the generation of the data")
    parser.add_argument("--standardize", type=int, default=1,
                        help="Standardize the features user by user at the generation of the data")

    parser.add_argument("--times", type=int, default=1, help="Number of simulations")
    parser.add_argument("--num_glob_iters", type=int, default=400, help="Number of communication rounds to plot")

    parser.add_argument("--nb_users", type=int, default=100, help="Number of all users for federated learning")
    parser.add_argument("--user_ratio", type=float, default=0.2,
                        help="Subsampling ratio for users at each communication round")
    parser.add_argument("--nb_samples", type=int, default=5000,
                        help="Number of all data points by user (conditionally to same_sample_size)")
    parser.add_argument("--sample_ratio", type=float, default=0.2,
                        help="Subsampling ratio for data points at each local update")
    parser.add_argument("--local_updates", type=int, default=100,
                        help="Number of local updates per selected user (local_epochs=local_updates*sample_ratio)")

    parser.add_argument("--weight_decay", type=float, default=5e-3, help="Regularization term")
    parser.add_argument("--local_learning_rate", type=float, default=10.0,
                        help="Learning rate multiplicative factor for local updates")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping value")

    parser.add_argument("--dp", type=str, default="None", choices=["None", "Gaussian"],
                        help="Differential Privacy or not")
    parser.add_argument("--epsilon_target", type=float, default=5.0, help="Epsilon parameter of differential privacy")

    parser.add_argument("--optimum", type=int, default=0,
                        help="True to train the model in a centralized setting (saves the best model)")
    parser.add_argument("--generate", type=int, default=0,
                        help="True to generate data (except Femnist)")
    parser.add_argument("--simulation", type=int, default=0,
                        help="True to run simulations (data needed)")
    parser.add_argument("--plot", type=int, default=0,
                        help="True to have plots (data needed)")

    args = parser.parse_args()

    run_simulation(dataset=args.dataset, number=args.number, dim_input=args.dim_input,
                   same_sample_size=args.same_sample_size, dim_output=args.dim_output,
                   nb_users=args.nb_users, user_ratio=args.user_ratio, nb_samples=args.nb_samples,
                   sample_ratio=args.sample_ratio,
                   local_updates=args.local_updates, weight_decay=args.weight_decay,
                   local_learning_rate=args.local_learning_rate,
                   max_norm=args.max_norm, dp=args.dp, epsilon_target=args.epsilon_target, normalise=args.normalise,
                   times=args.times, standardize=args.standardize,
                   optimum=args.optimum, num_glob_iters=args.num_glob_iters,
                   generate=args.generate, simulation=args.simulation, plot=args.plot)
