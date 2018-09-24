import cv2


class FrameGeneratorInterface:
    def __init__(self, video_source):
        pass

    def get_frame_size(self):
        pass

    def frames(self):
        pass

    def release(self):
        pass


class OpenCvFrameGenerator(FrameGeneratorInterface):
    def __init__(self, video_source):
        FrameGeneratorInterface.__init__(self, video_source)
        self.capture = cv2.VideoCapture(video_source)
        if not self.capture.isOpened():
            print('not self.capture.isOpened()')
        self._width = int(self.capture.get(3))
        self._height = int(self.capture.get(4))

    def get_frame_size(self):
        return self._width, self._height

    def frames(self):  # TODO rewrite
        while self.capture.isOpened():
            is_ok, frame = self.capture.read()
            if is_ok:
                yield frame
            else:
                break
        self.release()

    def release(self):
        self.capture.release()
        # print('self.capture.release()')
