#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

L = 0.1
N_POINTS = 500

def SE(x1, x2, sigma_f=1, l=L):
    exp_arg = -0.5 * np.dot((x2 - x1).T / l**2, x2 - x1)
    return sigma_f * np.exp(exp_arg)

def periodic_SE(x1, x2, period=1, sigma_f=1, l=L):
    dist = np.linalg.norm(x2 - x1)
    exp_arg = np.sin(np.pi * dist / (period)) ** 2 / l
    return sigma_f * np.exp(-exp_arg)

def matern32(x1, x2, sigma_f=1, l=L):
    dist = np.linalg.norm(x2 - x1)
    exp_arg = -np.sqrt(3) * dist / l
    matern_term = 1 + np.sqrt(3) * dist / l
    return sigma_f * matern_term * np.exp(exp_arg)

def matern52(x1, x2, sigma_f=1, l=L):
    dist = np.linalg.norm(x2 - x1)
    exp_arg = -np.sqrt(5) * dist / l
    matern_term = 1 + np.sqrt(5) * dist / l + \
        5 * dist ** 2 / (3 * l ** 2)
    return sigma_f * matern_term * np.exp(exp_arg)

def sample_from_random_function(ts, cov):
    """
    Evaluate a random C-infinity smooth function at points specified
    by xs. Hyperparameter l specifies how quickly nearby points
    decorrelate, and sigma_f specifies the variance of the function.
    """
    out_rows = ts.shape[0]
    cov_matrix = np.zeros((out_rows, out_rows))
    for i, x1 in enumerate(ts):
        for j, x2 in enumerate(ts):
            cov_matrix[i][j] = cov(x1, x2)
    mean = np.zeros(ts.shape)
    ys = np.random.multivariate_normal(mean, cov_matrix)
    return ys

def main():
    ts = np.linspace(-1, 1, N_POINTS)
    fig, ax = plt.subplots()
    for cov, name in [(SE, "RBF"), (periodic_SE, "Periodic RBF"), (matern32, "Matern 3/2"), (matern52, "Matern 5/2")]:
        ys = sample_from_random_function(ts, cov)
        ax.plot(ts, ys, label=name)
    ax.legend(loc="upper left")
    plt.show()
        

if __name__ == "__main__":
    main()
