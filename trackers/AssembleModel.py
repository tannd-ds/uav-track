import torch

from .association import AssociationModel
from .detection import DetectionModel, Detection


class AssembleModel:
    def __init__(self, detector: DetectionModel, associator: AssociationModel):
        self.detector = detector
        self.associator = associator
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def track(self, image, **kwargs) -> list[Detection]:
        frame_id = kwargs.get('frame_id', 0)
        sequence_name = kwargs.get('sequence_name', '')
        dets = self.detector.detect(image=image, frame_id=frame_id, sequence_name=sequence_name)
        tracks = self.associator.associate(dets)
        return tracks