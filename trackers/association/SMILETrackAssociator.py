from .base import AssociationModel
from .smiletrack.tracker.mc_SMILEtrack import SMILEtrack


class SMILETrackAssociator(AssociationModel):
    def __init__(self, args):
        model = SMILEtrack(args)
        super().__init__(model)

    def associate(self, detections, **kwargs):
        frame = kwargs.get("img")
        dets = [det.as_list for det in detections]
        track_objects = self.model.update(dets, frame)
        return track_objects
