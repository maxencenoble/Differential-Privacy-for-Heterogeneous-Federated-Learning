import emnist
import numpy as np
from sklearn.decomposition import PCA
from tqdm import trange
import random
import json
import os
import argparse
from os.path import dirname
from torchvision import datasets, transforms


def generate_data(similarity, num_users=50, num_samples=1000, number=0):
    """
    generate CIFAR-10 data among num_users users with a certain similarity
    :param similarity: portion of similar data between users. Float between 0 to 1
    :param num_users: number of users where data distributed among (int)
    :param num_samples: number of samples distributed to each user (int)
    :param number : number of dataset considered

    Remark : training ratio is just above 80%
    """

    assert num_users * num_samples == 50000 and 10000 % num_users == 0, "Distribution of nb users/samples not adapted"
    # Creation of directory
    root_path = os.path.dirname(__file__)
    print(root_path)
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

    data_transform = transforms.Compose([
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = datasets.CIFAR10(root=root_path, train=True, download=True, transform=data_transform)
    # SIZE : 50000
    test_set = datasets.CIFAR10(root=root_path, train=False, download=True, transform=data_transform)
    # SIZE : 10000

    train_images = np.array([train_set.__getitem__(i)[0].numpy() for i in range(50000)])
    train_labels=np.array([train_set.__getitem__(i)[1] for i in range(50000)])
    train_images = train_images.astype(np.float32)
    train_labels = train_labels.astype(np.int64)

    num_of_labels = len(set(train_labels))
    # = 10 (since there are 10 balanced classes for CIFAR-10)

    test_images = np.array([test_set.__getitem__(i)[0].numpy() for i in range(10000)])
    test_labels = np.array([test_set.__getitem__(i)[1] for i in range(10000)])
    test_images = test_images.astype(np.float32)
    test_labels = test_labels.astype(np.int64)

    cifar_data = []
    for i in range(min(train_labels), num_of_labels + min(train_labels)):
        idx = train_labels == i
        cifar_data.append(train_images[idx])

    iid_samples = int(similarity * num_samples)
    X_train = [[] for _ in range(num_users)]
    y_train = [[] for _ in range(num_users)]
    idx = np.zeros(num_of_labels, dtype=np.int64)

    # fill users data by labels to create dissimilarity
    for user in range(num_users):
        label = user % num_of_labels
        X_train[user] += cifar_data[label][idx[label]:idx[label] + num_samples - iid_samples].tolist()
        y_train[user] += (label * np.ones(num_samples - iid_samples)).tolist()
        idx[label] += num_samples - iid_samples

    print("Distribution of labels over the classes with the non similarity process (expected unbalanced) :")
    print("More weight for the first labels")
    print(idx)

    # create similarity% of iid data
    for user in range(num_users):
        labels = np.random.randint(0, num_of_labels, iid_samples)
        for label in labels:
            while idx[label] >= len(cifar_data[label]):
                label = (label + 1) % num_of_labels
            X_train[user].append(cifar_data[label][idx[label]].tolist())
            y_train[user] += (label * np.ones(1)).tolist()
            idx[label] += 1

    print("Distribution of the labels over the classes with the similarity process (expected balanced) :")
    print("In expectation : ", round(num_users * iid_samples / num_of_labels, 1), " for each label")
    print(idx)

    # create test data
    X_test = test_images.tolist()
    y_test = test_labels.tolist()

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in trange(num_users, ncols=120):
        uname = 'f_{0:07d}'.format(i)
        combined = list(zip(X_train[i], y_train[i]))
        random.shuffle(combined)
        X_train[i][:], y_train[i][:] = zip(*combined)
        train_len = len(X_train[i])
        test_len = len(test_images) // num_users
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X_train[i], 'y': y_train[i]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_test[test_len * i:test_len * (i + 1)],
                                         'y': y_test[test_len * i:test_len * (i + 1)]}
        test_data['num_samples'].append(test_len)

    print("Nb of train samples per user : ", train_data['num_samples'][0])
    print("Global nb of train samples : ", sum(train_data['num_samples']))
    print("Nb of test samples per user : ", test_data['num_samples'][0])
    print("Global nb of test samples : ", sum(test_data['num_samples']))

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity", type=float, default=0.1)
    parser.add_argument("--num_users", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()
    generate_data(similarity=args.similarity, num_users=args.num_users, num_samples=args.num_samples)
