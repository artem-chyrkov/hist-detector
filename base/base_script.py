from base.frame_generator import FrameGeneratorInterface
import cv2


class BaseScript:
    def __init__(self, frame_generator: FrameGeneratorInterface, delay=30):
        self.frame_generator = frame_generator
        self.delay = delay

    def run(self):
        self.init_callback(self.frame_generator.get_frame_size())
        for frame in self.frame_generator.frames():
            self.frame_callback(frame)
            key = cv2.waitKey(self.delay)
            if self.key_callback(key) != 0:
                self.frame_generator.release()  # TODO refactor
                break
        return 0

    def init_callback(self, frame_size):
        pass

    def frame_callback(self, frame):
        pass

    def key_callback(self, key):
        """
        :param key: key code
        :return: 0 if the cycle must be continued, 1 otherwise
        """
        if key & 0xFF == 27:
            return 1
        return 0


class FpsCounter:
    def __init__(self):
        self.tick_frequency = cv2.getTickFrequency()
        self.tick_previous = cv2.getTickCount()
        self.tick_current = 0

    def get_fps(self):
        self.tick_current = cv2.getTickCount()
        fps = self.tick_frequency / (self.tick_current - self.tick_previous)
        self.tick_previous = self.tick_current
        return fps
