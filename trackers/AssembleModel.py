import torch
from .detection import DetectionModel
from .association import AssociationModel

class AssembleModel:
    def __init__(self, detector: DetectionModel, associator: AssociationModel):
        self.detector = detector
        self.associator = associator
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def track(self, image, **kwargs):
        dets = self.detector.detect(image)
        tracks = self.associator.associate(dets)
        return tracks