import matplotlib.pyplot as plt
import matplotlib
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import os
from pathlib import Path

plt.rcParams.update({'font.size': 14})


def read_from_results(file_name):
    hf = h5py.File(file_name, 'r')
    string = file_name.split('_')
    if "norms" in string:
        rs_param_norms = np.array(hf.get('rs_param_norms')[:])
        if "SCAFFOLD" in string or "SCAFFOLD-warm" in string:
            rs_control_norms = np.array(hf.get('rs_control_norms')[:])
            return rs_param_norms, rs_control_norms
        else:
            return rs_param_norms

    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    rs_test_loss = np.array(hf.get('rs_test_loss')[:])
    rs_train_diss = np.array(hf.get('rs_train_diss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc, rs_test_loss, rs_train_diss


def get_all_training_data_value(num_glob_iters, algorithm, dataset, times, similarity, noise, number, dp,
                                sigma_gaussian, local_updates, sample_ratio, model_name):
    train_acc = np.zeros((times, num_glob_iters))
    train_loss = np.zeros((times, num_glob_iters))
    glob_acc = np.zeros((times, num_glob_iters))
    test_loss = np.zeros((times, num_glob_iters))
    train_diss = np.zeros((times, num_glob_iters))

    file_name = "./results/" + model_name + "/" + dataset + "_" + number + '_' + algorithm
    file_name += "_" + str(similarity) + "s"
    file_name += "_" + str(int(local_updates * sample_ratio)) + "K"
    if dp != "None":
        file_name += "_" + str(sigma_gaussian) + dp
    if noise:
        file_name += '_noisy'

    for i in range(times):
        f = file_name + "_" + str(i) + ".h5"
        train_acc[i, :], train_loss[i, :], glob_acc[i, :], test_loss[i, :], train_diss[i, :] = np.array(
            read_from_results(f))[:, :num_glob_iters]
    return glob_acc, train_acc, train_loss, test_loss, train_diss


def average_data(num_glob_iters, algorithm, dataset, times, similarity, noise, number, dp, sigma_gaussian,
                 local_updates, sample_ratio, model_name, cross_validation=False, k_fold=None, nb_fold=None,
                 local_learning_rate=None):
    glob_acc, train_acc, train_loss, test_loss, train_diss = get_all_training_data_value(
        num_glob_iters, algorithm,
        dataset, times,
        similarity,
        noise, number, dp, sigma_gaussian, local_updates, sample_ratio, model_name)

    glob_acc_data = np.average(glob_acc, axis=0)
    train_acc_data = np.average(train_acc, axis=0)
    train_loss_data = np.average(train_loss, axis=0)
    test_loss_data = np.average(test_loss, axis=0)
    train_diss_data = np.average(train_diss, axis=0)

    max_accuracy = []
    for i in range(times):
        max_accuracy.append(glob_acc[i].max())
    print("std Max Accuracy:", np.std(max_accuracy))
    mean_test_accuracy = np.mean(max_accuracy)
    print("Mean Max Accuracy:", np.mean(mean_test_accuracy))

    if cross_validation:
        assert local_learning_rate is not None and k_fold is not None, "Error on cross validation parameter"
        file_name = "./results_tuning/" + model_name + "/" + dataset + '_' + algorithm
        file_name += "_" + str(similarity) + "s"
        file_name += "_" + str(int(local_updates * sample_ratio)) + "K"
        if dp != "None":
            file_name += "_" + str(sigma_gaussian) + dp
        if noise:
            file_name += '_noisy'
        file_name += ".txt"

        dir_path = os.path.dirname(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = open(file_name, "a+")
        f.write(
            'Hyperparameter : {} - Test Accuracy at round {}/{} : {}\n'.format(local_learning_rate, k_fold + 1, nb_fold,
                                                                               mean_test_accuracy))
        f.close()
    else:
        # store average value to h5 file
        file_name = "./results/" + model_name + "/" + dataset + "_" + number + '_' + algorithm
        file_name += "_" + str(similarity) + "s"
        file_name += "_" + str(int(local_updates * sample_ratio)) + "K"
        if dp != "None":
            file_name += "_" + str(sigma_gaussian) + dp
        if noise:
            file_name += '_noisy'
        file_name += "_avg.h5"

        if len(glob_acc) != 0 & len(train_acc) & len(train_loss) & len(test_loss):
            with h5py.File(file_name, 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=glob_acc_data)
                hf.create_dataset('rs_train_acc', data=train_acc_data)
                hf.create_dataset('rs_train_loss', data=train_loss_data)
                hf.create_dataset('rs_test_loss', data=test_loss_data)
                hf.create_dataset('rs_train_diss', data=train_diss_data)
                hf.close()
        return 0


def get_all_norms(num_glob_iters, algorithm, dataset, times, similarity, noise, number, dp, sigma_gaussian,
                  local_updates, sample_ratio, model_name):
    file_name = "./results/" + model_name + "/" + dataset + "_" + number + '_' + algorithm + "_norms"
    file_name += "_" + str(similarity) + "s"
    file_name += "_" + str(int(local_updates * sample_ratio)) + "K"
    if dp != "None":
        file_name += "_" + str(sigma_gaussian) + dp
    if noise:
        file_name += '_noisy'

    param_norms = np.zeros((times, num_glob_iters))

    if algorithm == "SCAFFOLD" or algorithm == "SCAFFOLD-warm":
        control_norms = np.zeros((times, num_glob_iters))
        for i in range(times):
            f = file_name + "_" + str(i) + ".h5"
            param_norms[i, :], control_norms[i, :] = np.array(read_from_results(f))[:, :num_glob_iters]
        return param_norms, control_norms
    else:
        for i in range(times):
            f = file_name + "_" + str(i) + ".h5"
            param_norms[i, :] = np.array(read_from_results(f))[:num_glob_iters]
        return param_norms


def average_norms(num_glob_iters, algorithm, dataset, times, similarity, noise, number, dp, sigma_gaussian,
                  local_updates, sample_ratio, model_name):
    # store average value to h5 file
    file_name = "./results/" + model_name + "/" + dataset + "_" + number + '_' + algorithm + "_norms"
    file_name += "_" + str(similarity) + "s"
    file_name += "_" + str(int(local_updates * sample_ratio)) + "K"
    if dp != "None":
        file_name += "_" + str(sigma_gaussian) + dp
    if noise:
        file_name += '_noisy'
    file_name += "_avg.h5"

    if algorithm == "SCAFFOLD" or algorithm == "SCAFFOLD-warm":
        param_norms, control_norms = get_all_norms(num_glob_iters, algorithm, dataset, times, similarity,
                                                   noise, number, dp, sigma_gaussian, local_updates, sample_ratio,
                                                   model_name)
        glob_param_norms = np.average(param_norms, axis=0)
        glob_control_norms = np.average(control_norms, axis=0)
        if len(glob_param_norms) & len(glob_control_norms):
            with h5py.File(file_name, 'w') as hf:
                hf.create_dataset('rs_param_norms', data=glob_param_norms)
                hf.create_dataset('rs_control_norms', data=glob_control_norms)
    else:
        param_norms = get_all_norms(num_glob_iters, algorithm, dataset, times, similarity, noise, number, dp,
                                    sigma_gaussian,
                                    local_updates, sample_ratio, model_name)
        glob_param_norms = np.average(param_norms, axis=0)
        if len(glob_param_norms) != 0:
            with h5py.File(file_name, 'w') as hf:
                hf.create_dataset('rs_param_norms', data=glob_param_norms)
                hf.close()


def plot_norms(dataset, algorithms, noises, similarities, number, sigma_gaussian, local_updates, sample_ratio,
               user_ratio, model_name):
    epochs = int(local_updates * sample_ratio)
    colours = ['g', 'orange', 'black', 'purple']
    fig, axs = plt.subplots(1, len(similarities), constrained_layout=True, sharey='all')
    # fig.suptitle(f"{dataset} - Sigma_g={sigma_gaussian} - Local epochs={epochs}")

    if len(similarities) == 1:
        axs = [axs]

    for k, similarity in enumerate(similarities):
        axs[k].set_xlabel("Nb of communication rounds")
        axs[k].set_ylabel("Average Max Deltas over selected users")
        axs[k].set_yscale('log')
        axs[k].grid()
        if np.size(similarity) < 2:
            axs[k].set_title(str(100 * similarity) + "% Similarity")
        else:
            alpha, beta = similarity
            if alpha < 0 and beta < 0:
                similarity = "iid"
                axs[k].set_title("IID data")
            else:
                axs[k].set_title("(alpha, beta) = " + str(similarity))

        for noise in noises:
            j = 0
            for _, algorithm in enumerate(algorithms):
                for dp in ["None", "Gaussian"]:
                    file_name = "./results/" + model_name + "/" + dataset + '_' + number
                    file_name += "_" + algorithm + "_norms"
                    file_name += "_" + str(similarity) + "s"
                    if algorithm == "FedSGD":
                        file_name += "_1K"
                    else:
                        file_name += "_" + str(epochs) + "K"
                    if dp != "None":
                        file_name += "_" + str(sigma_gaussian) + dp
                    label = algorithm
                    color = colours[j]
                    if noise:
                        file_name += '_noisy'
                        label += ' with noise'
                        color += ':'
                    file_name += "_avg.h5"
                    if algorithm == "SCAFFOLD" or algorithm == "SCAFFOLD-warm":
                        param_norms, control_norms = np.array(read_from_results(file_name))[:, :]
                        if algorithm == "SCAFFOLD-warm":
                            param_norms[:round(4 / user_ratio)] = np.NaN
                            control_norms[:round(4 / user_ratio)] = np.NaN
                        if dp == "None":
                            axs[k].plot(param_norms, color=color, linestyle='dashed', label=label + ' (x)',
                                        alpha=0.6)
                        else:
                            label = "DP-" + label
                            axs[k].plot(param_norms, color, label=label + ' (x)')
                        color = colours[j + 1]
                        if dp == "None":
                            axs[k].plot(control_norms, color=color, linestyle='dashed', label=label + ' (c)',
                                        alpha=0.6)
                        else:
                            label = "DP-" + label
                            axs[k].plot(control_norms, color, label=label + ' (c)')
                    else:
                        param_norms = np.array(read_from_results(file_name))[:]
                        if dp == "None":
                            axs[k].plot(param_norms, color=color, linestyle='dashed', label=label, alpha=0.6)
                        else:
                            label = "DP-" + label
                            axs[k].plot(param_norms, color, label=label)
                    axs[k].legend(loc="lower left")
                j += 1
                if algorithm == "SCAFFOLD" or algorithm == "SCAFFOLD-warm":
                    j += 1
    plt.show()


def plot_train_dissimilarity(dataset, algorithms, noises, similarities, number, sigma_gaussian, local_updates,
                             sample_ratio, model_name):
    epochs = int(local_updates * sample_ratio)
    colours = ['g', 'orange', 'black']
    fig, axs = plt.subplots(1, len(similarities), constrained_layout=True, sharey='all')
    # fig.suptitle(f"{dataset} - Sigma_g={sigma_gaussian} - Local epochs={epochs}")

    if len(similarities) == 1:
        axs = [axs]

    for k, similarity in enumerate(similarities):
        axs[k].set_xlabel("Nb of communication rounds")
        axs[k].set_ylabel("Train Gradient Dissimilarity over all users")
        axs[k].grid()
        if np.size(similarity) < 2:
            axs[k].set_title(str(100 * similarity) + "% Similarity")
        else:
            alpha, beta = similarity
            if alpha < 0 and beta < 0:
                similarity = "iid"
                axs[k].set_title("IID data")
            else:
                axs[k].set_title("(alpha, beta) = " + str(similarity))

        for noise in noises:
            for j, algorithm in enumerate(algorithms):
                for dp in ["None", "Gaussian"]:
                    file_name = "./results/" + model_name + "/" + dataset + '_' + number
                    file_name += "_" + algorithm
                    file_name += "_" + str(similarity) + "s"
                    if algorithm == "FedSGD":
                        file_name += "_1K"
                    else:
                        file_name += "_" + str(epochs) + "K"
                    if dp != "None":
                        file_name += "_" + str(sigma_gaussian) + dp
                    label = algorithm
                    color = colours[j]
                    if noise:
                        file_name += '_noisy'
                        label += ' with noise'
                        color += ':'
                    file_name += "_avg.h5"
                    train_acc, train_loss, glob_acc, test_loss, train_diss = np.array(
                        read_from_results(file_name))[:, :]
                    if dp == "None":
                        axs[k].plot(train_diss, color=color, linestyle='dashed', label=label, alpha=0.6)
                    else:
                        label = "DP-" + label
                        axs[k].plot(train_diss, color, label=label)
                    axs[k].legend(loc="lower left")
    plt.show()


def plot_test_accuracy(dataset, algorithms, noises, similarities, number, sigma_gaussian, local_updates, sample_ratio,
                       model_name):
    epochs = int(local_updates * sample_ratio)
    colours = ['g', 'orange', 'black']
    fig, axs = plt.subplots(1, len(similarities), constrained_layout=True, sharey='all')
    # fig.suptitle(f"{dataset} - Sigma_g={sigma_gaussian} - Local epochs={epochs}")

    if len(similarities) == 1:
        axs = [axs]

    for k, similarity in enumerate(similarities):
        axs[k].set_xlabel("Nb of communication rounds")
        axs[k].set_ylabel("Test Accuracy over all users")
        axs[k].grid()
        if np.size(similarity) < 2:
            axs[k].set_title(str(100 * similarity) + "% Similarity")
        else:
            alpha, beta = similarity
            if alpha < 0 and beta < 0:
                similarity = "iid"
                axs[k].set_title("IID data")
            else:
                axs[k].set_title("(alpha, beta) = " + str(similarity))

        for noise in noises:
            for j, algorithm in enumerate(algorithms):
                for dp in ["None", "Gaussian"]:
                    file_name = "./results/" + model_name + "/" + dataset + '_' + number
                    file_name += "_" + algorithm
                    file_name += "_" + str(similarity) + "s"
                    if algorithm == "FedSGD":
                        file_name += "_1K"
                    else:
                        file_name += "_" + str(epochs) + "K"
                    if dp != "None":
                        file_name += "_" + str(sigma_gaussian) + dp
                    label = algorithm
                    color = colours[j]
                    if noise:
                        file_name += '_noisy'
                        label += ' with noise'
                        color += ':'
                    file_name += "_avg.h5"
                    train_acc, train_loss, glob_acc, test_loss, train_diss = np.array(
                        read_from_results(file_name))[:, :]
                    if dp == "None":
                        axs[k].plot(glob_acc, color=color, linestyle='dashed', label=label, alpha=0.6)
                    else:
                        label = "DP-" + label
                        axs[k].plot(glob_acc, color, label=label)
                    axs[k].legend(loc="lower right")
    plt.show()


def plot_test_accuracy_multiple_K(dataset, algorithms, noises, similarities, number, list_of_sigma_gaussian,
                                  list_of_local_updates, sample_ratio, model_name):
    list_of_epochs = [int(local_updates * sample_ratio) for local_updates in list_of_local_updates]
    list_of_linestyle = ['dotted', 'dashed', 'solid']
    colours = ['g', 'black']
    fig, axs = plt.subplots(1, len(similarities), constrained_layout=True, sharey='all')
    # fig.suptitle(f"{dataset} - Sigma_g={sigma_gaussian} - Local epochs={epochs}")

    if len(similarities) == 1:
        axs = [axs]

    for k, similarity in enumerate(similarities):
        axs[k].set_xlabel("Nb of communication rounds")
        axs[k].set_ylabel("Test Accuracy over all users")
        axs[k].grid()
        if np.size(similarity) < 2:
            axs[k].set_title(str(100 * similarity) + "% Similarity")
        else:
            alpha, beta = similarity
            if alpha < 0 and beta < 0:
                similarity = "iid"
                axs[k].set_title("IID data")
            else:
                axs[k].set_title("(alpha, beta) = " + str(similarity))

        for noise in noises:
            for j, algorithm in enumerate(algorithms):
                for dp in ["Gaussian"]:
                    for (epochs, sigma_gaussian, linestyle) in list(
                            zip(list_of_epochs, list_of_sigma_gaussian, list_of_linestyle)):
                        file_name = "./results/" + model_name + "/" + dataset + '_' + number
                        file_name += "_" + algorithm
                        file_name += "_" + str(similarity) + "s"
                        file_name += "_" + str(epochs) + "K"
                        file_name += "_" + str(sigma_gaussian) + dp
                        label = "DP-" + algorithm
                        color = colours[j]
                        if noise:
                            file_name += '_noisy'
                            label += ' with noise'
                            color += ':'
                        file_name += "_avg.h5"
                        train_acc, train_loss, glob_acc, test_loss, train_diss = np.array(
                            read_from_results(file_name))[:, :]
                        str_K = ", K={}".format(int(epochs / sample_ratio))
                        axs[k].plot(glob_acc, color=color, linestyle=linestyle, label=label + str_K)
                        axs[k].legend(loc="lower right")
    plt.show()


def plot_train_loss(dataset, algorithms, noises, similarities, number, sigma_gaussian, local_updates, sample_ratio,
                    model_name):
    epochs = int(local_updates * sample_ratio)
    colours = ['g', 'orange', 'black']
    fig, axs = plt.subplots(1, len(similarities), constrained_layout=True, sharey='all')
    # fig.suptitle(f"{dataset} - Sigma_g={sigma_gaussian} - Local epochs={epochs}")

    if len(similarities) == 1:
        axs = [axs]

    for k, similarity in enumerate(similarities):
        axs[k].set_xlabel("Nb of communication rounds")
        axs[k].grid()
        log = True
        if np.size(similarity) < 2:
            axs[k].set_ylabel("log(F(x_t)-F(x*))")
            axs[k].set_title(str(100 * similarity) + "% Similarity")
        else:
            alpha, beta = similarity
            if alpha < 0 and beta < 0:
                similarity = "iid"
                axs[k].set_title("IID data")
                axs[k].set_ylabel("F(x_t)")
                log = False
            else:
                axs[k].set_title("(alpha, beta) = " + str(similarity))
                axs[k].set_ylabel("log(F(x_t)-F(x*))")

        for noise in noises:
            for j, algorithm in enumerate(algorithms):
                for dp in ["None", "Gaussian"]:
                    file_name = "./results/" + model_name + "/" + dataset + '_' + number
                    file_name += "_" + algorithm
                    file_name += "_" + str(similarity) + "s"
                    if algorithm == "FedSGD":
                        file_name += "_1K"
                    else:
                        file_name += "_" + str(epochs) + "K"
                    if dp != "None":
                        file_name += "_" + str(sigma_gaussian) + dp
                    label = algorithm
                    color = colours[j]
                    if noise:
                        file_name += '_noisy'
                        label += ' with noise'
                        color += ':'
                    file_name += "_avg.h5"
                    train_acc, train_loss, glob_acc, test_loss, train_diss = np.array(
                        read_from_results(file_name))[:, :]
                    if log:
                        train_loss = np.log(train_loss)
                    if dp == "None":
                        axs[k].plot(train_loss, color=color, linestyle='dashed', label=label, alpha=0.6)
                    else:
                        label = "DP-" + label
                        axs[k].plot(train_loss, color, label=label)
                    axs[k].legend(loc="lower left")
    plt.show()


def plot_train_loss_multiple_K(dataset, algorithms, noises, similarities, number, list_of_sigma_gaussian,
                               list_of_local_updates, sample_ratio, model_name):
    list_of_epochs = [int(local_updates * sample_ratio) for local_updates in list_of_local_updates]
    list_of_linestyle = ['dotted', 'dashed', 'solid']
    colours = ['g', 'black']
    fig, axs = plt.subplots(1, len(similarities), constrained_layout=True, sharey='all')
    # fig.suptitle(f"{dataset} - Sigma_g={sigma_gaussian} - Local epochs={epochs}")

    if len(similarities) == 1:
        axs = [axs]

    for k, similarity in enumerate(similarities):
        axs[k].set_xlabel("Nb of communication rounds")
        axs[k].grid()
        log = True
        if np.size(similarity) < 2:
            axs[k].set_ylabel("log(F(x_t)-F(x*))")
            axs[k].set_title(str(100 * similarity) + "% Similarity")
        else:
            alpha, beta = similarity
            if alpha < 0 and beta < 0:
                similarity = "iid"
                axs[k].set_title("IID data")
                axs[k].set_ylabel("F(x_t)")
                log = False
            else:
                axs[k].set_title("(alpha, beta) = " + str(similarity))
                axs[k].set_ylabel("log(F(x_t)-F(x*))")

        for noise in noises:
            for j, algorithm in enumerate(algorithms):
                for dp in ["Gaussian"]:
                    for (epochs, sigma_gaussian, linestyle) in list(
                            zip(list_of_epochs, list_of_sigma_gaussian, list_of_linestyle)):
                        file_name = "./results/" + model_name + "/" + dataset + '_' + number
                        file_name += "_" + algorithm
                        file_name += "_" + str(similarity) + "s"
                        file_name += "_" + str(epochs) + "K"
                        file_name += "_" + str(sigma_gaussian) + dp
                        label = "DP-" + algorithm
                        color = colours[j]
                        if noise:
                            file_name += '_noisy'
                            label += ' with noise'
                            color += ':'
                        file_name += "_avg.h5"
                        train_acc, train_loss, glob_acc, test_loss, train_diss = np.array(
                            read_from_results(file_name))[:, :]
                        if log:
                            train_loss = np.log(train_loss)
                        str_K = ", K={}".format(int(epochs / sample_ratio))
                        axs[k].plot(train_loss, color=color, linestyle=linestyle, label=label + str_K)
                        axs[k].legend(loc="lower left")
    plt.show()
