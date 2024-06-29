from ultralytics import RTDETR

from .detection import Detection
from .utils.yolo import yolo_results_to_base_detection


class RTDETRModel:
    def __init__(self, weights_path):
        super().__init__()
        self.model = RTDETR(weights_path)

    def track(self, img_dir, **kwargs) -> list[Detection]:
        persist = kwargs.get("persist", True)
        tracker = kwargs.get("tracker", "botsort.yaml")
        device = kwargs.get("device", "cpu")
        verbose = kwargs.get("verbose", True)
        results = self.model.track(img_dir, persist=persist, tracker=tracker, device=device, verbose=verbose)
        dets = yolo_results_to_base_detection(results)
        return dets
