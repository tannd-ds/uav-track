from ultralytics import YOLO
from .base import ObjectModel
from .detection import Detection

class YOLOModel(ObjectModel):
    def __init__(self, weights_path):
        super().__init__()
        self.model = YOLO(weights_path)

    def detect(self):
        # Since we use YOLO, we don't need to get detections manually
        pass

    def track(self, img_dir, **kwargs) -> list[Detection]:
        persist = kwargs.get("persist", True)
        tracker = kwargs.get("tracker", "botsort.yaml")
        device = kwargs.get("device", "cpu")
        verbose = kwargs.get("verbose", True)
        results = self.model.track(img_dir, persist=persist, tracker=tracker, device=device, verbose=verbose)
        dets = []

        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id = box.cls.cpu().numpy()[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            det = Detection(int(box.id.item()))
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.track_id = int(box.id.item())

            dets.append(det)
        return dets

    def draw_predictions(self, frame, boxes):
        # YOLO-specific drawing logic (same as your original code)
        pass