class SlidingStripeRois:
    def __init__(self):
        self.rows = 0
        self.rois = []
        self.current_roi_row = -1

    def init(self, frame_width, frame_height, roi_width, roi_height):
        cols = frame_width // roi_width
        self.rows = frame_height // roi_height
        self.rois.clear()
        for i in range(self.rows):
            self.rois.append([(j * roi_width, i * roi_height, roi_width, roi_height) for j in range(cols)])
        self.current_roi_row = -1

    def get_current_rois(self):
        self.current_roi_row += 1
        if self.current_roi_row == self.rows:
            self.current_roi_row = 0
        return self.rois[self.current_roi_row]

    def count(self):
        return self.rows * len(self.rois[0])
