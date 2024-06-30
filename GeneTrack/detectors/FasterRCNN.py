import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor

from . import DetectorBase, Detection


class FasterRCNNDetector(DetectorBase):
    def __init__(self, weights_path=None, num_classes=10):
        model = self.get_model(num_classes)
        super().__init__(model, weights_path)

        self.load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image):
        if isinstance(image, np.ndarray):
            image = [self.numpy_to_tensor(image)]

        with torch.no_grad():
            predictions = self.model(image)[0]

        conf = predictions['scores'].cpu().numpy()
        cls = predictions['labels'].cpu().numpy()
        # turn boxes from xyxy to xywh
        boxes = predictions['boxes'].cpu().numpy()
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2

        dets = Detection(conf=conf, xywh=boxes, cls=cls)
        return dets

    @staticmethod
    def get_model(num_classes):
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def load_weights(self, weights_path):
        # Note: Your model weights should be saved like this: `torch.save(model.state_dict(), save_path)`
        weights = torch.load(weights_path)
        self.model.load_state_dict(weights)
