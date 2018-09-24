from math import ceil, floor
# from numpy import mean, std, reshape
# from matplotlib import pyplot as plt


""" Suspicion level constants """  # TODO move out
SL_NON_SUSPICIOUS = 0
SL_UNKNOWN = 1
SL_PROBABLY_SUSPICIOUS = 2
SL_SUSPICIOUS = 3


def calculate_mean_and_dispersion_i(hist):
    m, m2, H, n = 0.0, 0.0, 0, len(hist)
    for i in range(n):
        H += hist[i]
        m += i * hist[i]
        m2 += i * i * hist[i]
    m /= H
    m2 /= H
    D = m2 - m * m
    return m, D


def calculate_mean_and_dispersion(hist, bin_centers):  # TODO refactor
    m, m2, H, n = 0.0, 0.0, 0, len(hist)
    for i in range(n):
        H += hist[i]
        m += bin_centers[i] * hist[i]
        m2 += bin_centers[i] * bin_centers[i] * hist[i]
    m /= H
    m2 /= H
    D = m2 - m * m
    return m, D


def percentile(x, alpha):
    y = x.copy()
    y.sort()
    i = alpha * (len(y) - 1)
    i_floor, i_ceil = int(floor(i)), int(ceil(i))
    if i_floor == i_ceil:
        return y[i_ceil]
    else:
        p_floor, p_ceil = i_ceil - i, i - i_floor
        return p_floor * y[i_floor] + p_ceil * y[i_ceil]


def get_bin_centers(bin_edges):
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    return bin_centers


def get_bin_index(e, bin_edges):
    N = len(bin_edges)
    for i in range(N - 1):
        if bin_edges[i] <= e < bin_edges[i + 1]:
            return i
    return N - 1
