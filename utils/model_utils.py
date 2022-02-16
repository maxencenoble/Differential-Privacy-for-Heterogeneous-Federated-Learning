import json
import numpy as np
import os
import torch
import torch.nn as nn


def read_data(dataset, number, similarity, dim_pca=None):
    """Parses data in given train and test data directories

    Assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Returns:
        users: list of user ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """

    train_data_dir = os.path.join('data', dataset, 'data', 'train')
    test_data_dir = os.path.join('data', dataset, 'data', 'test')
    users = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    if dim_pca is not None:
        train_files = [f for f in train_files if
                       f.endswith(number + '_' + similarity + '_' + 'pca' + str(dim_pca) + '.json')]
    else:
        train_files = [f for f in train_files if f.endswith(number + '_' + similarity + '.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        users.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    if dim_pca is not None:
        test_files = [f for f in test_files if
                      f.endswith(number + '_' + similarity + '_' + 'pca' + str(dim_pca) + '.json')]
    else:
        test_files = [f for f in test_files if f.endswith(number + '_' + similarity + '.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    users = list(sorted(train_data.keys()))

    return users, groups, train_data, test_data


def read_data_cross_validation(dataset, number, similarity, k_fold, nb_fold, dim_pca):
    assert k_fold in np.arange(nb_fold), "Index of cross validation not correct"

    """Parses data in given train directories to conduct cross validation

    Assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Returns:
        users: list of user ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_data_dir = os.path.join('data', dataset, 'data', 'train')
    users = []
    groups = []
    all_data = {}
    train_data = {}
    test_data = {}

    print(dim_pca)

    train_files = os.listdir(train_data_dir)
    if dim_pca is not None:
        train_files = [f for f in train_files if
                       f.endswith(number + '_' + similarity + '_' + 'pca' + str(dim_pca) + '.json')]
    else:
        train_files = [f for f in train_files if f.endswith(number + '_' + similarity + '.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        users.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        all_data.update(cdata['user_data'])

    users = list(sorted(all_data.keys()))
    train_len = len(all_data[users[0]]['x'])

    for index in range(len(users)):
        id = users[index]
        train_data[id] = {
            'x': all_data[id]['x'][:round(k_fold * train_len / nb_fold)] + all_data[id]['x'][
                                                                           round((k_fold + 1) * train_len / nb_fold):],
            'y': all_data[id]['y'][:round(k_fold * train_len / nb_fold)] + all_data[id]['y'][
                                                                           round((k_fold + 1) * train_len / nb_fold):]}
        test_data[id] = {
            'x': all_data[id]['x'][round(k_fold * train_len / nb_fold):round((k_fold + 1) * train_len / nb_fold)],
            'y': all_data[id]['y'][round(k_fold * train_len / nb_fold):round((k_fold + 1) * train_len / nb_fold)]}

    return users, groups, train_data, test_data


def read_user_data(index, data, dataset):
    """Returns:
        id: id of user
        train_data: list of (data, labels) for training
        test_data: list of (data, labels) for testing
    """
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if dataset == "CIFAR-10":
        X_train = torch.Tensor(X_train).view(-1, 3, 32, 32).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, 3, 32, 32).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    else:
        # image flattened for FEMNIST, MNIST
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data
