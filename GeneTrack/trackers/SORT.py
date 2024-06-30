import numpy as np

from .STrack import STrack
from .basetrack import TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH


class SORT(object):
    def __init__(self, args):
        self.tracked_stracks = []

        self.frame_id = 0
        self.args = args
        self.kalman_filter = self.get_kalmanfilter()

    def update(self, results, img=None):
        """Updates object tracker with new detections and returns tracked object bounding boxes."""
        self.frame_id += 1

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        detections = self.init_track(bboxes, scores, cls, img)

        # Predict the current location with KF for all tracks
        self.multi_predict(self.tracked_stracks)

        # IoU-based matching (linear assignment)
        dists = matching.iou_distance(self.tracked_stracks, detections)
        matches, unmatched_tracks, unmatched_detections = matching.linear_assignment(dists,
                                                                                     thresh=self.args.match_thresh)

        # Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracked_stracks[track_idx].update(detections[detection_idx], self.frame_id)

        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks:
            self.tracked_stracks[track_idx].mark_lost()

        # Create and activate new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            track = detections[detection_idx]
            track.activate(self.kalman_filter, self.frame_id)
            self.tracked_stracks.append(track)

        # Keep only tracked (not lost or removed) tracks
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        return np.asarray([x.result for x in self.tracked_stracks], dtype=np.float32)

    @staticmethod
    def init_track(dets, scores, cls, img=None):
        """Initialize object tracking with detections and scores using STrack algorithm."""
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # detections

    @staticmethod
    def multi_predict(tracks):
        """Returns the predicted tracks using the YOLOv8 network."""
        STrack.multi_predict(tracks)

    @staticmethod
    def get_kalmanfilter():
        """Returns a Kalman filter object for tracking bounding boxes."""
        return KalmanFilterXYAH()
