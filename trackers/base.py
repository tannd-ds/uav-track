class ObjectModel:
    def detect(self):
        raise NotImplementedError

    def track(self, img_dir, **kwargs):
        raise NotImplementedError

    def draw_predictions(self, frame, boxes):
        raise NotImplementedError