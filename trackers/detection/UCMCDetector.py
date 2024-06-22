import cv2
import numpy as np
from ultralytics import YOLO
from trackers.detection.ucmc.detector.mapper import Mapper
from trackers.detection.base import DetectionModel
from . import Detection

class UCMCDetector(DetectionModel):
    def __init__(self,
                 cam_para_file:str,
                 weights_path:str
                 ):
        super().__init__()
        self.seq_length = 0
        self.gmc = None

        self.load(cam_para_file, weights_path)

    def load(self, cam_para_file, weights_path):
        self.mapper = Mapper(cam_para_file, "MOT17")
        self.model = YOLO(weights_path)

    def get_dets(self, img, conf_thresh=0):
        dets = []

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.model(frame, imgsz=512, device="0")

        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id = box.cls.cpu().numpy()[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 10 and h <= 10 or conf <= conf_thresh:
                continue

            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])
            det_id += 1

            dets.append(det)
        return dets

    def detect(self, image):
        return self.get_dets(image)
