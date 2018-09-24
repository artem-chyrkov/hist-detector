from numpy import inf, argmax
from base.statistics import calculate_mean_and_dispersion_i


def algorithm_2(hist):
    n = len(hist)
    separate_index, mode1, mode2, f = 3, -1, -1, inf
    start_index = separate_index
    for si in range(start_index, n - start_index):
        hist_left = hist[:si]
        hist_right = hist[si:]
        m_left, D_left = calculate_mean_and_dispersion_i(hist_left)
        m_right, D_right = calculate_mean_and_dispersion_i(hist_right)
        fi = D_left + D_right
        if fi < f:
            f = fi
            m_left = argmax(hist_left)
            m_right = argmax(hist_right)
            separate_index, mode1, mode2 = si, m_left, m_right + si
    return mode1, mode2, separate_index
