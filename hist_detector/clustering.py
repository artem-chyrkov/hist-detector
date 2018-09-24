import numpy as np
import cv2
from sklearn.cluster import AffinityPropagation
from base.statistics import percentile


class Clustering:
    def __init__(self):
        self.fast_feature_detector = cv2.FastFeatureDetector_create(threshold=25)
        self.clustering = None
        self.non_target_clusters = []

    def calculate_feature_vector(self, target):
        patch_gray, suspicion_level, additional_data = target
        m1, m2, D = additional_data
        keypoints = self.fast_feature_detector.detect(patch_gray)
        return [len(keypoints), m1, m2, D, suspicion_level]

    def fit(self, targets):
        """
        :param targets: { (patch_gray, suspicion_level, additional_data) }
        :return:
        """
        features = [self.calculate_feature_vector(target) for target in targets]
        self.clustering = AffinityPropagation().fit(features)
        self.non_target_clusters.clear()
        non_target_cluster_indices = self.get_non_target_cluster_indices(alpha_percentile=0.25)
        # print('non_target_cluster_indices:', non_target_cluster_indices)
        self.non_target_clusters.extend(non_target_cluster_indices)

    def is_target(self, targets_to_be: list):
        """
        :return: boolean array
        """
        feature_vectors = [self.calculate_feature_vector((frame_roi, suspicion_level,
                                                          additional_data))
                           for roi, frame_roi, suspicion_level, additional_data in targets_to_be]
        patch_indices = self.clustering.predict(feature_vectors)
        is_target_patches = []
        is_target_patches += [i in self.non_target_clusters for i in patch_indices]
        return is_target_patches

    def get_cluster_count(self):
        return len(self.clustering.cluster_centers_indices_)

    def get_non_target_cluster_indices(self, alpha_percentile=0.25):
        keypoint_counts = np.transpose(self.clustering.cluster_centers_)[0].astype(int)
        kc_threshold = int(round(percentile(keypoint_counts, alpha_percentile)))
        # print('kc_threshold ==', kc_threshold)
        non_target_indices = [i for i in range(len(keypoint_counts))
                              if keypoint_counts[i] <= kc_threshold]
        return non_target_indices
