import numpy as np
from scipy.special import gammaln, logsumexp
import math

# Meta parameters
# T: nb of communication rounds
# K: nb of local updates
# M: nb of users
# R: nb of data points used for training

T = 100
K = 50
M = 100
R = int(0.8 * 5000)

# 0.8 : training ratio

# delta: privacy parameter
# l: user subsampling ratio
# s: data subsampling ratio

delta = 1 / (M * R)
l = 0.2
s = 0.2

# sigma_gaussian: standard deviation of Gaussian noise used in the algorithm
# sigma_g: standard deviation of Gaussian noise "evaluated" for privacy towards a third party

sigma_gaussian = 60.0
sigma_gaussian_actual = sigma_gaussian * np.sqrt(l * M)


# The privacy parameter epsilon is calculated for any third party who has access to the last iterate of the algorithm
# Our method consists of a minimization problem over the variable `alpha` from the RDP bound
# We notably use the upper bound for subsampling provided in Theorem 9 in https://arxiv.org/pdf/1808.00087.pdf

# Remark that our framework is only available for mechanisms with eps(infinity)=+inf !
# (verified for Gaussian mechanisms and its compositions)

def logcomb(n, k):
    """Returns the logarithm of comb(n,k)"""
    return (gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1))


def RDP_epsilon_bound_gaussian(alpha):
    """Returns the epsilon RDP bound for Gaussian mechanism with std parameter sigma_gaussian_actual"""
    return 0.5 * alpha / (sigma_gaussian_actual ** 2)


def cgf_subsampling_for_int_alpha(alpha: int, eps_func, sub_ratio):
    """
    Parameters:
    :param alpha: int, >1
    :param eps_func: fun(float->float), epsilon RDP bound evaluation function
    :param sub_ratio: subsampling ratio

    Returns a tight upper bound of the CGF(alpha) for the s-subsampled eps_func(alpha),
    ie (alpha-1)*eps_subsampled(alpha)"""
    alpha = int(alpha)
    log_moment_two = 2 * np.log(sub_ratio) + logcomb(alpha, 2) + np.minimum(
        np.log(4) + eps_func(2.) + np.log(1 - np.exp(-eps_func(2.))), eps_func(2.) + np.log(2))
    log_moment_j = lambda j: np.log(2) + (j - 1) * eps_func(j) + j * np.log(sub_ratio) + logcomb(alpha, j)
    all_log_moments_j = [log_moment_j(j) for j in range(3, alpha + 1, 1)]
    return logsumexp([0, log_moment_two] + all_log_moments_j)


def intermediate_epsilon_rdp_bound_for_int_alpha(alpha: int):
    """
    Parameters:
    :param alpha: int, >1

    Returns an upper RDP epsilon bound after K composed s-subsampled Gaussian mechanisms.
    """
    return K * cgf_subsampling_for_int_alpha(alpha, RDP_epsilon_bound_gaussian, s) / (alpha - 1)


def epsilon_rdp_bound_for_int_alpha(alpha: int):
    """
    Parameters:
    :param alpha: int, >1

    Returns an upper RDP epsilon bound after T composed l-subsampled [K composed s-subsampled Gaussian mechanisms].
    """
    return T * cgf_subsampling_for_int_alpha(alpha, intermediate_epsilon_rdp_bound_for_int_alpha, l) / (alpha - 1)


def epsilon_rdp_bound_for_float_alpha(alpha: float):
    """
    Parameters:
    :param alpha: float, >1

    Returns an upper RDP epsilon bound after T composed l-subsampled [K composed s-subsampled Gaussian mechanisms],
    using linear interpolation on the CGF (by convexity) to approximate the bound.
    """
    floor_alpha = math.floor(alpha)
    ceil_alpha = math.ceil(alpha)
    if floor_alpha == 1:
        first = 0.
    else:
        first = (1 - alpha + floor_alpha) * (floor_alpha - 1) * epsilon_rdp_bound_for_int_alpha(floor_alpha) / (
                alpha - 1)
    second = (alpha - floor_alpha) * (ceil_alpha - 1) * epsilon_rdp_bound_for_int_alpha(ceil_alpha) / (alpha - 1)
    return first + second


def epsilon_dp_bound_for_int_alpha(alpha: int):
    """
    Parameters:
    :param alpha: int, >1

    Returns an upper DP epsilon bound after T composed l-susampled [K composed s-subsampled Gaussian mechanisms].
    """
    return epsilon_rdp_bound_for_int_alpha(alpha) + np.log(1 / delta) / (alpha - 1)


def epsilon_dp_bound_for_float_alpha(alpha: float):
    """
    Parameters:
    :param alpha: float, >1

    Returns an upper DP epsilon bound after T composed l-susampled [K composed s-subsampled Gaussian mechanisms].
    """
    return epsilon_rdp_bound_for_float_alpha(alpha) + np.log(1 / delta) / (alpha - 1)


if __name__ == "__main__":

    # Parameters to tune by hand:
    # alpha_int_max: int
    # n_points: int

    # 1. Determine the integer alpha with the best DP bound (grid search between 2 and alpha_int_max)
    alpha_int_max = 100
    alpha_int_space = np.arange(2, alpha_int_max + 1, 1)
    argmin_int = np.argmin([epsilon_dp_bound_for_int_alpha(alpha_int) for alpha_int in alpha_int_space])
    alpha_int_min = alpha_int_space[argmin_int]
    if alpha_int_min == alpha_int_max:
        print("Increase alpha_int_max!")

    alpha_lower = alpha_int_min - 1. + 0.0001  # instability around alpha=1
    alpha_upper = alpha_int_min + 1.

    # 2. Determine the float alpha with the best DP bound (grid search around alpha_int_min: +-1)
    n_points = 1000  # precision of the grid
    alpha_float_space = np.linspace(alpha_lower, alpha_upper, n_points)
    idx_min = np.argmin([epsilon_dp_bound_for_float_alpha(alpha_float) for alpha_float in alpha_float_space])
    alpha_float_min = alpha_float_space[idx_min]
    print("Best epsilon DP bound:{:.4f}".format(epsilon_dp_bound_for_float_alpha(alpha_float_min)))
