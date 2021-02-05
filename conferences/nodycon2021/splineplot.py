#!/usr/bin/env python3

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import wrapper

N_POINTS = 300
SMOOTHNESS = 5
N_KNOTS = 25

def plotfunc(ts):
    return np.exp(ts) * (ts<0) - np.exp(-ts) * (ts>0)
    ## return 2 * ts * np.exp(-0.25*ts**2)

def smoothing_spline(ts, ys, smoothness):
    sort_indices = np.argsort(ts)
    return scipy.interpolate.UnivariateSpline(ts[sort_indices], ys[sort_indices], s=smoothness)

def cardinal_bspline(ts, ys, knots):
    sort_indices = np.argsort(ts)
    return scipy.interpolate.LSQUnivariateSpline(ts[sort_indices], ys[sort_indices], knots)

def freeknot_splines(ts, ys):
    return wrapper.barsN(ts, ys, burnin=5000, prior_param=(0, 80), iknots=N_KNOTS)

def main():
    test_ts = np.linspace(-10, 10, N_POINTS*4)
    ts = np.random.uniform(-10, 10, N_POINTS)
    ys = plotfunc(ts) + np.random.normal(0, 0.1, size=ts.size)

    knots = np.linspace(-10, 10, N_KNOTS + 2)[1:-1]
    smoothing = smoothing_spline(ts, ys, SMOOTHNESS)
    cardinal = cardinal_bspline(ts, ys, knots)
    bars = freeknot_splines(ts, ys)
    

    fig, ax = plt.subplots()
    ax.scatter(ts, ys, label="Observed data", s=1)
    ax.plot(test_ts, plotfunc(test_ts), "k--", label="True function", alpha=0.5)
    ax.plot(test_ts, smoothing(test_ts), label="Smoothing spline", alpha=0.5)
    ax.plot(test_ts, cardinal(test_ts), label="Cardinal BSpline", alpha=0.5)
    ax.plot(test_ts, bars(test_ts), label="Bayesian free-knot splines")
    ax.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    main()
