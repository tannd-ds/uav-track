from .base import AssociationModel
from .ucmc.tracker.ucmc import UCMCTrack

class UCMCAssociator(AssociationModel):
    def __init__(self):
        model = UCMCTrack(100.0, 100.0, 5, 5, 10, 10.0, 10, "MOT", 0.5, False, None)
        super().__init__(model)


    def associate(self, detections, **kwargs):
        frame_id = kwargs.get('frame_id')
        self.model.update(detections, frame_id=frame_id)
        return detections
