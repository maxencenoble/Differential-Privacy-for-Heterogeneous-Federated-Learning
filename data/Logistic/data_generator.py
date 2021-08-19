#!/usr/bin/env python
import numpy as np
import json
import random
import os
import matplotlib.pyplot as plt


def logit(X, W, b):
    res = np.dot(X, W) + b
    res = np.exp(res) / np.sum(np.exp(res))
    return int(np.argmax(res))


def generate_data(num_users=100, same_sample_size=True, num_samples=20, dim_input=40, dim_output=10,
                  noise_ratio=0.05, alpha=0., beta=0., ratio_training=0.8, number=0, normalise=False,
                  standardize=False, iid=False):
    """
        generate Logistic-regression data among num_users users
        :param num_users : number of users where data is distributed among (int)
        :param same_sample_size : determines if the users have the same sample size (boolean)
        :param num_samples : number of samples distributed to each user according to same_sample_size (int)
        :param dim_input : dimension of input (int)
        :param dim_output : nb of classes (int)
        :param noise_ratio : probability of noise in the exact labels (float between 0 and 1)
        :param alpha : parameter of similarity between users (i.i.d if ==0)
        :param beta : parameter of similarity among one user's data (i.i.d if ==0)
        :param ratio_training : ratio of training samples over all samples
        :param number : id of dataset considered (if identical datasets are generated)
        :param normalise : normalise inputs by point
        :param standardize : standardize inputs by user
        :param iid : generates iid data (alpha and beta useless)
    """

    similarity = str((alpha, beta))

    if iid:
        similarity = "iid"

    # Creation of directory
    root_path = os.path.dirname(__file__)
    train_path = root_path + '/data/train/mytrain_' + str(number) + '_' + similarity + '.json'
    test_path = root_path + '/data/test/mytest_' + str(number) + '_' + similarity + '.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # For consistent results
    np.random.seed(0)

    # Sanity check
    assert (
            num_users > 0 and num_samples > 0 and dim_input > 0 and dim_output > 0 and noise_ratio >= 0)

    X_split = [[] for _ in range(num_users)]  # X for each user
    y_split = [[] for _ in range(num_users)]  # y for each user

    if not same_sample_size:
        # Find users' sample sizes based on the power law (heterogeneity)
        samples_per_user = np.random.lognormal(num_samples ** (1 / 4), 1, num_users).astype(int) + num_samples
        print(samples_per_user)
    else:
        samples_per_user = (num_samples * np.ones(num_users)).astype(int)

    indices_per_user = np.insert(samples_per_user.cumsum(), 0, 0, 0).astype(int)
    num_total_samples = indices_per_user[-1].astype(int)

    if not iid:
        # Each user's mean is drawn from N(b_k, 1)
        # where b_k is drawn from N(0, beta) -> (i.i.d. data only if beta=0)
        pre_mean_X = np.array([np.sqrt(beta) * np.random.randn(dim_input) for _ in range(num_users)])
        mean_X = pre_mean_X + np.random.randn(num_users, dim_input)
    else:
        common_mean_X = np.random.randn(dim_input)
        mean_X = np.array([common_mean_X for _ in range(num_users)])

    # Covariance matrix for X
    Sigma = np.zeros((dim_input, dim_input))
    diag = np.array([(j + 1) ** (-1.2) for j in range(dim_input)])
    np.fill_diagonal(Sigma, diag)

    # Generate weights
    if not iid:
        mean_W = np.array([np.sqrt(alpha) * np.random.randn(dim_input, dim_output) for _ in range(num_users)])
        W_total = mean_W + np.random.randn(num_users, dim_input, dim_output)

        mean_b = np.sqrt(alpha) * np.random.randn(num_users, dim_output)
        b_total = mean_b + np.random.randn(num_users, dim_output)
    else:
        common_mean_W = np.random.randn(dim_input, dim_output)
        W_total = np.array([common_mean_W for _ in range(num_users)])
        common_mean_b = np.random.randn(dim_output)
        b_total = np.array([common_mean_b for _ in range(num_users)])

    # Keep all users' inputs and labels in one array,
    # indexed according to indices_per_user.
    #   (e.g. X_total[indices_per_user[n]:indices_per_user[n+1], :] = X_n)
    #   (e.g. y_total[indices_per_user[n]:indices_per_user[n+1]] = y_n)
    X_total = np.zeros((num_total_samples, dim_input))
    y_total = np.zeros(num_total_samples)

    for n in range(num_users):
        # Generate data
        X_n = np.random.multivariate_normal(mean_X[n], Sigma, samples_per_user[n])
        X_total[indices_per_user[n]:indices_per_user[n + 1], :] = X_n
        y_total[indices_per_user[n]:indices_per_user[n + 1]] = [logit(X_sample, W_total[n], b_total[n]) for X_sample in
                                                                X_n]

    # Apply noise: randomly flip some of y_n with probability noise_ratio
    noises = np.random.binomial(1, noise_ratio, num_total_samples)
    new_classes = np.random.randint(0, dim_output, num_total_samples)
    y_total = [int(new_c) if noise == 1 else y for (new_c, noise, y) in zip(new_classes, noises, y_total)]

    # Save each user's data separately
    for n in range(num_users):
        X_n = X_total[indices_per_user[n]:indices_per_user[n + 1], :]
        y_n = y_total[indices_per_user[n]:indices_per_user[n + 1]]
        X_split[n] = X_n.tolist()
        y_split[n] = y_n

    print("=" * 80)
    print("Generated synthetic data for logistic regression successfully.")
    print("    Total # users       : {}".format(num_users))
    print("    Input dimension     : {}".format(dim_input))
    print("    Nb of classes       : {}".format(dim_output))
    print("    Total # of samples  : {}".format(num_total_samples))
    print("    Minimum # of samples: {}".format(np.min(samples_per_user)))
    print("    Maximum # of samples: {}".format(np.max(samples_per_user)))
    print("=" * 80)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    all_train_len = []

    for i in range(num_users):
        uname = 'f_{0:07d}'.format(i)
        combined = list(zip(X_split[i], y_split[i]))
        random.shuffle(combined)
        X_split[i][:], y_split[i][:] = zip(*combined)
        num_samples = len(X_split[i])
        train_len = int(ratio_training * num_samples)
        all_train_len.append(train_len)
        test_len = num_samples - train_len
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X_split[i][:train_len], 'y': y_split[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_split[i][train_len:], 'y': y_split[i][train_len:]}
        test_data['num_samples'].append(test_len)

    if standardize:
        print("=" * 80)
        print("Standardizing features by user dataset ...")
        for i in range(num_users):
            uname = 'f_{0:07d}'.format(i)
            for dim in range(dim_input):
                max_dim = max(X_split[i][:all_train_len[i]][dim])
                min_dim = min(X_split[i][:all_train_len[i]][dim])
                for input_train in train_data['user_data'][uname]['x']:
                    input_train[dim] = (input_train[dim] - min_dim) / (max_dim - min_dim)
                for input_test in test_data['user_data'][uname]['x']:
                    input_test[dim] = (input_test[dim] - min_dim) / (max_dim - min_dim)
        print("=" * 80)

    if normalise:
        print("=" * 80)
        print("Normalising every point...")
        for i in range(num_users):
            uname = 'f_{0:07d}'.format(i)
            for i in range(len(train_data['user_data'][uname]['x'])):
                input_train = train_data['user_data'][uname]['x'][i]
                norm_2 = np.sqrt(np.sum(np.array(input_train) ** 2))
                train_data['user_data'][uname]['x'][i] = list(map(lambda x: x / norm_2, input_train))
            for i in range(len(test_data['user_data'][uname]['x'])):
                input_test = test_data['user_data'][uname]['x'][i]
                norm_2 = np.sqrt(np.sum(np.array(input_test) ** 2))
                test_data['user_data'][uname]['x'][i] = list(map(lambda x: x / norm_2, input_test))
        print("=" * 80)

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)

    print("=" * 120)
    print("Saved all users' data sucessfully.")
    print("    Train path:", os.path.join(os.curdir, train_path))
    print("    Test path :", os.path.join(os.curdir, test_path))
    print("=" * 120)


def main():
    generate_data(num_users=100, same_sample_size=True, num_samples=500, dim_input=40, dim_output=10,
                  noise_ratio=0.05, alpha=0., beta=0., ratio_training=0.8, number=0, normalise=False, standardize=False)


if __name__ == '__main__':
    main()
