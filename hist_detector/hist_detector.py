from math import sqrt, fabs
from base.statistics import SL_NON_SUSPICIOUS, SL_UNKNOWN, SL_PROBABLY_SUSPICIOUS, SL_SUSPICIOUS

import cv2
import numpy as np
# from matplotlib import pyplot as plt

from base import mixture_separation
from base.ring_list import RingList
from base.statistics import *


class HistDetector:
    def __init__(self, bins_count, alpha_D4, k_1_alpha, k_2_alpha, q_alpha):
        # constants:
        self.bins_count = bins_count
        self.alpha_D4 = alpha_D4
        self.k_1_alpha = k_1_alpha
        self.k_2_alpha = k_2_alpha
        self.q_alpha = q_alpha

        # fields:
        self.D1 = 0.0
        self.delta_m = 0.0
        self.n_T = 0.0
        self.D4 = 0.0
        self.P_T = RingList()  # (m, D) of textures
        self.P_F = RingList()  # D of fragments
        # self.P_PS = RingList()  # D of probably suspicious
        self.hist_m_PT = []
        self.bin_edges_m_PT = []

    def init(self, ring_list_size):
        """ init """
        self.P_T.init(ring_list_size)
        self.P_F.init(ring_list_size)
        # self.P_PS.init(ring_list_size)
        self._clear()

    def _clear(self):
        self.P_T.clear()
        self.P_F.clear()
        # self.P_PS.clear()
        self.D1, self.delta_m, self.n_T, self.D4 = 0.0, 0.0, 0.0, 0.0

    def train(self, patch_params):
        """
        :param patch_params: { (m, D) for all patches }
        """
        self._clear()
        m, D = np.reshape(patch_params, (2, len(patch_params)))
        N = len(D)
        # D = exclude_abnormal(D)
        # m, D = exclude_abnormal_2dimension(m, D)

        # pre-calculate D1
        hist_D, bin_edges_D = np.histogram(D, int(sqrt(N)) + 1)
        bin_centers_D = get_bin_centers(bin_edges_D)
        m1, m2, si = mixture_separation.algorithm_2(hist_D)
        mD, _ = calculate_mean_and_dispersion(hist_D[:si], bin_centers_D[:si])
        self.D1 = self.k_1_alpha * mD
        # plt.plot(bin_centers_D, hist_D)
        # plt.show()

        # pre-calculate D4
        self.D4 = percentile(D, self.alpha_D4)

        # fill P_T and P_F:
        for i in range(N):
            if D[i] < self.D1:  # [0..D1)
                self.P_T.add((m[i], D[i]))
            elif D[i] < self.D4:  # [D1..D4)
                self.P_F.add(D[i])
            # else:  # [D4; +inf)
            #     self.P_PS.add(D[i])

        self.recalculate_thresholds()

    def detect(self, frame, rois):
        """
        :param frame: BGR frame
        :param rois:
        :return: { (roi, suspicion_level, additional_data) }
                   where additional_data == (m1, m2, D)
        """
        target = []  # see :return
        for roi in rois:
            x, y, w, h = roi
            patch_bgr = frame[y: y + h, x: x + w]
            patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
            hist = self.calculate_histogram(patch_gray)
            m, D = calculate_mean_and_dispersion_i(hist)
            if D >= self.D4:  # [D4; +inf)
                # self.P_PS.add(D)
                target.append((roi, SL_PROBABLY_SUSPICIOUS, (m, m, D)))
                continue
            if D < self.D1:  # [0; D1)
                self.P_T.add((m, D))
                # target.append((roi, SL_NON_SUSPICIOUS))
                continue
            m1, m2, _ = mixture_separation.algorithm_2(hist)
            if fabs(m2 - m1) <= self.k_2_alpha * sqrt(D):  # |m1 - m2| < dm
                self.P_F.add(D)
                target.append((roi, SL_UNKNOWN, (m1, m2, D)))
                # continue
            else:  # check each mode
                if self.is_frequent_mode(m1) and self.is_frequent_mode(m2):  # mixture of known
                    # target.append((roi, SL_NON_SUSPICIOUS))
                    pass
                else:
                    target.append((roi, SL_SUSPICIOUS, (m1, m2, D)))

        self.recalculate_thresholds()
        return target

    def is_patch_still_suspicious(self, patch_bgr):  # TODO consider refactoring possibility
        """
        Checks if patch is still suspicious (for repass approach)
        :return: SL_NON_SUSPICIOUS / SL_UNKNOWN / SL_PROBABLY_SUSPICIOUS / SL_SUSPICIOUS
        """
        patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        hist = self.calculate_histogram(patch_gray)
        m, D = calculate_mean_and_dispersion_i(hist)
        if D >= self.D4:
            return SL_PROBABLY_SUSPICIOUS
        elif D < self.D1:
            return SL_NON_SUSPICIOUS
        else:
            m1, m2, _ = mixture_separation.algorithm_2(hist)
            if fabs(m2 - m1) <= self.k_2_alpha * sqrt(D):
                return SL_UNKNOWN
            else:
                if self.is_frequent_mode(m1) and self.is_frequent_mode(m2):
                    return SL_NON_SUSPICIOUS
                else:
                    return SL_SUSPICIOUS

    def is_frequent_mode(self, mi):
        """
        :param mi: i-th mode value (m1 or m2)
        :return: True/False
        """
        if not (self.bin_edges_m_PT[0] < mi < self.bin_edges_m_PT[len(self.bin_edges_m_PT) - 1]):
            return False
        mi_bin_index = get_bin_index(mi, self.bin_edges_m_PT)
        mi_frequency = self.hist_m_PT[mi_bin_index]
        return mi_frequency > self.n_T

    def recalculate_thresholds(self):
        """ recalculate_thresholds """
        m_PT, D_PT = [], []  # TODO optimize
        for i in range(self.P_T.get_actual_size()):
            m, D = self.P_T.get()[i]
            m_PT.append(m)
            D_PT.append(D)

        # calculate D1
        self.D1 = np.max(D_PT)
        # plt.plot(get_bin_centers(hist_edges_D), hist_D)
        # bi = get_bin_index(mD, hist_edges_D)
        # plt.scatter([mD], [hist_D[bi]])
        # plt.show()

        # calculate n_T
        self.hist_m_PT, self.bin_edges_m_PT = np.histogram(m_PT, int(sqrt(len(m_PT))) + 1)
        self.n_T = self.q_alpha * np.mean(self.hist_m_PT)
        # plt.plot(self.hist_m_PT)
        # plt.show()

        # calculate D4
        # self.D4 = percentile(D_PT + self.P_F.get() + self.P_PS.get(), self.alpha_D4)
        self.D4 = np.max(D_PT + self.P_F.get())

        # print('\nself.P_T: i=', self.P_T.get_index_to_replace(), self.P_T.get_actual_size(), ' /',
        #       self.P_T._max_size)
        # print('self.P_F: i=', self.P_F.get_index_to_replace(), self.P_F.get_actual_size(), ' /',
        #       self.P_F._max_size)

    def calculate_histogram(self, patch_gray):
        """
        :param patch_gray: grayscale patch
        :return: histogram of self.bins_count with int values
        """
        hist = cv2.calcHist([patch_gray], [0], None, [self.bins_count], [0, 256])
        hist = np.reshape(hist, (self.bins_count,)).astype(int)
        return hist
