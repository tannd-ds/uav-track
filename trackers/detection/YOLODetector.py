import torch
from ultralytics import YOLO
from .base import DetectionModel

class YOLODetector(DetectionModel):
    def __init__(self, weights_path:str):
        super().__init__()
        self.model = YOLO(weights_path)

    def detect(self, image):
        return self.model(image, device=self.device)