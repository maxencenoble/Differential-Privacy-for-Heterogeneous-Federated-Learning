import numpy as np
import scipy.special as scp
from scipy.optimize import minimize
import math

T = 250
K = 50
M = 40
R = int(0.8*2500)
# 0.8 : training ratio

delta = 1/(M*R)
l = 0.2
s = 0.1

sigma_g = 10.0


# The privacy is calculated for any third party who has access to the last iterate

def epsilon_prime_integer_gaussian(alpha):
    """Assuming alpha INTEGER >1 : returns the RDP upper bound for the subsampled Gaussian mechanism."""
    sum = 1 + (s ** 2) * scp.comb(alpha, 2) * np.min(
        [4 * (np.exp(1 / ((sigma_g * np.sqrt(l * M)) ** 2)) - 1), 2 * np.exp(1 / ((sigma_g * np.sqrt(l * M)) ** 2))])
    for j in range(3, alpha + 1):
        sum += 2 * (s ** j) * scp.comb(alpha, j) * np.exp(
            (j - 1) * 0.5 * j / ((sigma_g * np.sqrt(l * M)) ** 2))
    return np.log(sum) / (alpha - 1)


def epsilon_prime_all_gaussian(alpha):
    """Assuming alpha REAL >1 : returns the RDP upper bound for the subsampled Gaussian mechanism."""
    floor_alpha = math.floor(alpha)
    ceil_alpha = math.ceil(alpha)
    if floor_alpha == 1:
        first = 0.
    else:
        first = (1 - alpha + floor_alpha) * (floor_alpha - 1) * epsilon_prime_integer_gaussian(floor_alpha) / (
                    alpha - 1)
    second = (alpha - floor_alpha) * (ceil_alpha - 1) * epsilon_prime_integer_gaussian(ceil_alpha) / (alpha - 1)
    return first + second


def epsilon_prime_integer_general(alpha):
    """Assuming alpha INTEGER >1 : RDP upper bound of user subsampling after K Gaussian mechanisms."""
    sum = 1 + (l ** 2) * scp.comb(alpha, 2) * np.min(
        [4 * (np.exp(K * epsilon_prime_integer_gaussian(2)) - 1),
         2 * np.exp(K * epsilon_prime_integer_gaussian(2))])
    for j in range(3, alpha + 1):
        sum += 2 * (l ** j) * scp.comb(alpha, j) * np.exp(
            (j - 1) * K * epsilon_prime_integer_gaussian(j))
    return np.log(sum) / (alpha - 1)


def epsilon_prime_all_general(alpha):
    """Assuming alpha REAL >1 : RDP upper bound of user subsampling after K Gaussian mechanisms."""
    floor_alpha = math.floor(alpha)
    ceil_alpha = math.ceil(alpha)
    if floor_alpha == 1:
        first = 0.
    else:
        first = (1 - alpha + floor_alpha) * (floor_alpha - 1) * epsilon_prime_integer_general(floor_alpha) / (alpha - 1)
    second = (alpha - floor_alpha) * (ceil_alpha - 1) * epsilon_prime_integer_general(ceil_alpha) / (alpha - 1)
    return first + second


def final_dp_bound(alpha):
    """Assuming alpha REAL >1 : DP upper bound of the whole mechanism."""
    return T * epsilon_prime_all_general(alpha) + np.log(1 / delta) / (alpha - 1)


if __name__ == "__main__":
    bnds = ((1.1, None),)
    res = minimize(lambda alpha: final_dp_bound(alpha), np.array([2.5]), bounds=bnds)
    print("Right alpha: ", res.x[0].round(3))
    print("Corresponding epsilon: ", res.fun[0].round(3))
