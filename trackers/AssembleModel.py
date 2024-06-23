import torch

from .association import AssociationModel
from .detection import DetectionModel, Detection


class AssembleModel:
    def __init__(self, detector: DetectionModel, associator: AssociationModel):
        self.detector = detector
        self.associator = associator
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def track(self, image, **kwargs) -> list[Detection]:
        dets = self.detector.detect(image)
        tracks = self.associator.associate(dets)
        return tracks