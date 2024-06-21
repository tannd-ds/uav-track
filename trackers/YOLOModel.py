from ultralytics import YOLO
from ultralytics. engine. results import Results
from .base import ObjectModel

class YOLOModel(ObjectModel):
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def update_detections(self):
        # Since we use YOLO, we don't need to update detections manually
        pass

    def track(self, img_dir, **kwargs) -> list[Results]:
        return self.model.track(img_dir, **kwargs)

    def draw_predictions(self, frame, boxes):
        # YOLO-specific drawing logic (same as your original code)
        pass