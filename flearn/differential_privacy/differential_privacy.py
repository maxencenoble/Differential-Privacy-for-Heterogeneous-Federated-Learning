import torch
import numpy as np


def GaussianMechanism(grad_tensor,sigma_g, max_norm, batch_size):
    """adds Gaussian noise to stochastic gradients
    :param grad_tensor : stands for a stochastic gradient
    :param sigma_g : variance of the Gaussian noise (defined by DP theory)
    :param max_norm : clipping value
    :param batch_size : nb of data point in the considered batch"""
    std_gaussian = (sigma_g*max_norm)/batch_size
    gaussian_noise = torch.normal(0, std_gaussian, grad_tensor.shape)
    return grad_tensor + gaussian_noise

