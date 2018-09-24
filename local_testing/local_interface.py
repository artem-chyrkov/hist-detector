from sys import path
path.append('../')

from base.frame_generator import OpenCvFrameGenerator
from base.base_script import BaseScript, FpsCounter
from base.statistics import SL_NON_SUSPICIOUS, SL_UNKNOWN, SL_PROBABLY_SUSPICIOUS, SL_SUSPICIOUS
from hist_detector.hist_detector_facade import HistDetectorFacade, STAGE_TRAINING, STAGE_DETECT
from local_testing.draw_patches import draw_patches

import cv2


class Application(BaseScript):
    def __init__(self, video_source):
        # BaseScript.__init__(self, OpenCvFrameGenerator(video_source))
        BaseScript.__init__(self, OpenCvFrameGenerator(video_source), delay=1)
        self.size = ''
        self.fps_counter = FpsCounter()
        self.hist_detector_facade = HistDetectorFacade()

    def init_callback(self, frame_size):
        self.size = '%dx%d' % frame_size
        self.hist_detector_facade.init(frame_size[0], frame_size[1])

    def frame_callback(self, frame):
        frame_to_show = frame.copy()
        self.hist_detector_facade.update(frame)
        self.show_results(frame_to_show)
        cv2.putText(frame_to_show, self.size, (1, 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
        cv2.putText(frame_to_show, 'FPS: %.2f' % self.fps_counter.get_fps(), (1, 29), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
        cv2.imshow('frame', frame_to_show)

    def key_callback(self, key):
        if key & 0xFF == 99:  # 'c'
            self.hist_detector_facade.fit_clustering()
            print('cluster_count ==', self.hist_detector_facade.clustering.get_cluster_count())
            return 0
        elif key & 0xFF == 100:  # 'd'
            self.hist_detector_facade.throw_out_non_targets_by_clustering()
            return 0
        elif key & 0xFF == 114:  # 'r'
            self.hist_detector_facade.throw_out_non_targets_by_repass()
            return 0
        return BaseScript.key_callback(self, key)

    def show_results(self, frame_to_show):
        hdf = self.hist_detector_facade

        if hdf.stage == STAGE_DETECT:
            patches_suspicious = []
            patches_probably_suspicious = []
            for roi, patch_bgr, suspicion_level, additional_data in hdf.target_patches:
                if suspicion_level == SL_SUSPICIOUS:
                    patches_suspicious += [patch_bgr]
                elif suspicion_level == SL_PROBABLY_SUSPICIOUS:
                    patches_probably_suspicious += [patch_bgr]

            draw_patches(patches_suspicious, 'suspicious')
            draw_patches(patches_probably_suspicious, 'probably_suspicious')
            # print('len(hdf.target_patches):', len(hdf.target_patches))

            if len(hdf.non_target_patches) > 0:
                draw_patches([patch_bgr
                              for roi, patch_bgr, suspicion_level, additional_data
                              in hdf.non_target_patches
                              if suspicion_level == SL_SUSPICIOUS
                              or suspicion_level == SL_PROBABLY_SUSPICIOUS], 'NON TARGETS')

        cv2.putText(frame_to_show,
                    'training' if hdf.stage == STAGE_TRAINING
                    else 'D1=%.2f; D4=%.2f' % (hdf.hist_detector.D1, hdf.hist_detector.D4),
                    (1, 43), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))


if __name__ == '__main__':
    VIDEO_FILENAME = '/insert/your/video.mp4'
    Application(VIDEO_FILENAME).run()
