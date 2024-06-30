from ultralytics import YOLO

from . import DetectorBase, Detection


class YOLODetector(DetectorBase):
    def __init__(self, weight_path: str):
        model = YOLO(weight_path)
        super().__init__(model=model)

    def detect(self, img) -> Detection:
        yolo_dets = self.model(img, verbose=self.verbose, device=self.device)[0].cpu().numpy()
        dets = Detection(yolo_dets.boxes.conf, yolo_dets.boxes.xywh, yolo_dets.boxes.cls)
        return dets

