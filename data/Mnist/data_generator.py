import emnist
import numpy as np
from sklearn.decomposition import PCA
from tqdm import trange
import random
import json
import os
import argparse
from os.path import dirname


def generate_data(similarity, num_users=100, num_samples=20, ratio_training=0.8, number=0, normalise=True):
    """
    generate MNIST data among num_users users with a certain similarity
    :param similarity: portion of similar data between users. Float between 0 to 1
    :param num_users: number of users where data distributed among (int)
    :param num_samples: number of samples distributed to each user (int)
    :param ratio_training: Float between 0 and 1
    :param number : number of dataset considered
    :param normalise : normalise inputs by point
    """
    # Creation of directory
    root_path = os.path.dirname(__file__)
    train_path = root_path + '/data/train/mytrain_' + str(number) + '_' + str(similarity) + '.json'
    test_path = root_path + '/data/test/mytest_' + str(number) + '_' + str(similarity) + '.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # For consistent results
    np.random.seed(0)

    # Sanity check
    assert (num_users > 0 and num_samples > 0 and similarity >= 0)

    # Creation of dataset
    dataset = 'digits'
    train_images, train_labels = emnist.extract_training_samples(mnist)
    train_images = np.reshape(train_images, (train_images.shape[0], -1))
    train_images = train_images.astype(np.float32)
    train_labels = train_labels.astype(np.int64)

    num_of_labels = len(set(train_labels))
    # = 10 (since there are 10 balanced classes for MNIST)

    emnist_data = []
    for i in range(min(train_labels), num_of_labels + min(train_labels)):
        idx = train_labels == i
        emnist_data.append(train_images[idx])

    iid_samples = int(similarity * num_samples)
    X = [[] for _ in range(num_users)]
    y = [[] for _ in range(num_users)]
    idx = np.zeros(num_of_labels, dtype=np.int64)

    # create %similarity of iid data
    for user in range(num_users):
        labels = np.random.randint(0, num_of_labels, iid_samples)
        for label in labels:
            # to prevent filling problems in case of similarity 1.0
            while idx[label] == 6000:
                label += 1
                label %= 10
            X[user].append(emnist_data[label][idx[label]].tolist())
            y[user] += (label * np.ones(1)).tolist()
            idx[label] += 1

    print("Distribution of the labels over the classes with the similarity process (expected balanced) :")
    print("In expectation : ", round(num_users * iid_samples / num_of_labels, 1), " for each label")
    print(idx)

    # fill remaining data
    # We consider successively every user and fill the the remaining data with one label, that is updated for each user
    for user in range(num_users):
        label = user % num_of_labels
        X[user] += emnist_data[label][idx[label]:idx[label] + num_samples - iid_samples].tolist()
        y[user] += (label * np.ones(num_samples - iid_samples)).tolist()
        idx[label] += num_samples - iid_samples

    print("Distribution of labels over the classes with the non similarity process (expected unbalanced) :")
    print("More weight for the first labels")
    print(idx)

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in trange(num_users, ncols=120):
        uname = 'f_{0:07d}'.format(i)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(ratio_training * num_samples)
        test_len = num_samples - train_len
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    print("Nb of train samples per user : ", train_data['num_samples'][0])
    print("Global nb of train samples : ", sum(train_data['num_samples']))
    print("Nb of test samples per user : ", test_data['num_samples'][0])
    print("Global nb of test samples : ", sum(test_data['num_samples']))

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


def generate_pca_data(similarity, dim_pca=60, num_users=100, num_samples=20, ratio_training=0.8, number=0,
                      normalise=True):
    """
    generate MNIST-Balanced data among num_users users with a certain similarity, projected along a family over dim_pca elements
    :param similarity: portion of similar data between users. Float between 0 to 1
    :param dim_pca: nb of components for PCA (int)
    :param num_users: number of users where data distributed among (int)
    :param num_samples: number of samples distributed to each user (int)
    :param ratio_training: Float between 0 and 1
    :param number : number of dataset considered
    :param normalise : normalise inputs by point
    """
    # Creation of directory
    root_path = os.path.dirname(__file__)
    train_path = root_path + '/data/train/mytrain_' + str(number) + '_' + str(similarity) + '_' + 'pca' + str(
        dim_pca) + '.json'
    test_path = root_path + '/data/test/mytest_' + str(number) + '_' + str(similarity) + '_' + 'pca' + str(
        dim_pca) + '.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # For consistent results
    np.random.seed(0)

    # Sanity check
    assert (num_users > 0 and num_samples > 0 and similarity >= 0 and dim_pca > 0)

    # Creation of dataset
    dataset = 'mnist'
    train_images, train_labels = emnist.extract_training_samples(dataset)
    train_images = np.reshape(train_images, (train_images.shape[0], -1))
    train_images = train_images.astype(np.float32)
    train_labels = train_labels.astype(np.int64)
    
    print("New dimension from PCA: ", dim_pca)
    # PCA from training set
    pca = PCA(n_components=dim_pca)
    pca.fit(train_images)
    print("Explained variance ratio for the first 20 directions", pca.explained_variance_ratio_[:20])
    train_images = pca.transform(train_images)

    num_of_labels = len(set(train_labels))
    # = 10 (since there are 47 balanced classes for EMNIST)

    emnist_data = []
    for i in range(min(train_labels), num_of_labels + min(train_labels)):
        idx = train_labels == i
        emnist_data.append(train_images[idx])

    iid_samples = int(similarity * num_samples)
    X = [[] for _ in range(num_users)]
    y = [[] for _ in range(num_users)]
    idx = np.zeros(num_of_labels, dtype=np.int64)

    # create %similarity of iid data
    for user in range(num_users):
        labels = np.random.randint(0, num_of_labels, iid_samples)
        for label in labels:
            # to prevent filling problems in case of similarity 1.0
            while idx[label] == 6000:
                label += 1
                label %= 10
            X[user].append(emnist_data[label][idx[label]].tolist())
            y[user] += (label * np.ones(1)).tolist()
            idx[label] += 1

    print("Distribution of the labels over the classes with the similarity process (expected balanced) :")
    print("In expectation : ", round(num_users * iid_samples / num_of_labels, 1), " for each label")
    print(idx)

    # fill remaining data
    # We consider successively every user and fill the the remaining data with one label, that is updated for each user
    for user in range(num_users):
        label = user % num_of_labels
        X[user] += emnist_data[label][idx[label]:idx[label] + num_samples - iid_samples].tolist()
        y[user] += (label * np.ones(num_samples - iid_samples)).tolist()
        idx[label] += num_samples - iid_samples

    print("Distribution of labels over the classes with the non similarity process (expected unbalanced) :")
    print("More weight for the first labels")
    print(idx)

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in trange(num_users, ncols=120):
        uname = 'f_{0:07d}'.format(i)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(ratio_training * num_samples)
        test_len = num_samples - train_len
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    print("Nb of train samples per user : ", train_data['num_samples'][0])
    print("Global nb of train samples : ", sum(train_data['num_samples']))
    print("Nb of test samples per user : ", test_data['num_samples'][0])
    print("Global nb of test samples : ", sum(test_data['num_samples']))

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity", type=float, default=0.4)
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    generate_data(similarity=args.similarity, num_users=args.num_users, num_samples=args.num_samples)
