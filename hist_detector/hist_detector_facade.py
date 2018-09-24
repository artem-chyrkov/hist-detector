from base.statistics import calculate_mean_and_dispersion_i
from base.statistics import SL_NON_SUSPICIOUS, SL_UNKNOWN, SL_PROBABLY_SUSPICIOUS, SL_SUSPICIOUS
from hist_detector.roi import SlidingStripeRois
from hist_detector.hist_detector import HistDetector
from hist_detector.clustering import Clustering
from cv2 import cvtColor, COLOR_BGR2GRAY


class Params:
    """
    Parameters of the HistDetector method
    """
    ROI_SIZE = 50
    TRAINING_FULL_FRAME_COUNT = 5
    RING_LIST_FULL_FRAME_COUNT = 30
    BINS_COUNT = 32
    ALPHA_D4 = 0.99
    K_1_ALPHA = 0.95  # multiplier for D1
    K_2_ALPHA = 2.5  # multiplier for delta_m
    Q_ALPHA = 0.3  # multiplier for n_T


STAGE_TRAINING = 0
STAGE_DETECT = 1


def get_frame_roi(frame, roi):
    x, y, w, h = roi
    return frame[y: y + h, x: x + w]


class HistDetectorFacade:
    """
    Adapter for more understandable HistDetector usage
    """
    def __init__(self):
        self.update = None  # <-- function pointer
        self.hist_detector = HistDetector(Params.BINS_COUNT, Params.ALPHA_D4,
                                          Params.K_1_ALPHA, Params.K_2_ALPHA, Params.Q_ALPHA)
        self.rois = SlidingStripeRois()
        self.target_patches = []  # { (roi, patch_bgr, suspicion_level) }
        self.non_target_patches = []  # to show non-targets for debug/demo purposes

        self.stage = STAGE_TRAINING
        self.training_frame_count = 0
        self.current_frame_index = 0
        self.patch_params = []  # { (m, D) for all patches }

        self.clustering = Clustering()

    def init(self, frame_width, frame_height):
        """ init """
        self.update = self._update_train  # <-- function pointer
        self.rois.init(frame_width, frame_height, Params.ROI_SIZE, Params.ROI_SIZE)
        self.hist_detector.init(Params.RING_LIST_FULL_FRAME_COUNT * self.rois.count())

        self.stage = STAGE_TRAINING
        self.training_frame_count = self.rois.rows * Params.TRAINING_FULL_FRAME_COUNT
        self.current_frame_index = 1
        self.patch_params.clear()
        self.target_patches.clear()
        self.non_target_patches.clear()

    def _update_train(self, frame):
        """ _update_train """
        if self.current_frame_index >= self.training_frame_count:
            self.hist_detector.train(self.patch_params)
            self.stage = STAGE_DETECT
            self.update = self._update_detect
            return
        self.current_frame_index += 1

        rois = self.rois.get_current_rois()
        for x, y, w, h in rois:
            patch_bgr = frame[y: y + h, x: x + w]
            patch_gray = cvtColor(patch_bgr, COLOR_BGR2GRAY)
            m, D = calculate_mean_and_dispersion_i(self.hist_detector.calculate_histogram(patch_gray))
            self.patch_params += [(m, D)]

    def _update_detect(self, frame):
        """ _update_detect """
        rois = self.rois.get_current_rois()
        target_rois = self.hist_detector.detect(frame, rois)
        new_target_patches = [(roi, get_frame_roi(frame, roi), suspicion_level, additional_data)
                              for roi, suspicion_level, additional_data in target_rois]
        self.target_patches.extend(new_target_patches)

    def fit_clustering(self):
        """ fit_clustering """
        target = [(cvtColor(patch_bgr, COLOR_BGR2GRAY), suspicion_level, additional_data)
                  for _, patch_bgr, suspicion_level, additional_data in self.target_patches]
        self.clustering.fit(target)

    def throw_out_non_targets_by_clustering(self):
        """ throw out non-targets by clustering """
        is_targets = self.clustering.is_target(self.target_patches)
        updated_target_patches = []
        self.non_target_patches.clear()
        for i in range(len(is_targets)):
            if is_targets[i]:
                updated_target_patches += [self.target_patches[i]]
            else:
                self.non_target_patches.append(self.target_patches[i])
        self.target_patches.clear()
        self.target_patches.extend(updated_target_patches)

    def throw_out_non_targets_by_repass(self):
        """ throw out non-targets by histogram detector repass """
        updated_target_patches = []
        self.non_target_patches.clear()
        for target_patch in self.target_patches:
            roi, patch_bgr, suspicion_level, additional_data = target_patch
            is_still_suspicious = self.hist_detector.is_patch_still_suspicious(patch_bgr)
            if is_still_suspicious in [SL_SUSPICIOUS, SL_PROBABLY_SUSPICIOUS]:
                updated_target_patches += [target_patch]
            else:
                self.non_target_patches.append(target_patch)
        self.target_patches.clear()
        self.target_patches.extend(updated_target_patches)
