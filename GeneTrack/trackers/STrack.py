import numpy as np
import torch

from .basetrack import BaseTrack, TrackState
from .utils.kalman_filter import KalmanFilterXYAH


def xywh2ltwh(x):
    """
    Convert the bounding box format from [x, y, w, h] to [x1, y1, w, h], where x1, y1 are the top-left coordinates.

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding box coordinates in the xywh format

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xyltwh format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y


class STrack(BaseTrack):
    """
    Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter that is used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.

    Methods:
        predict(): Predict the next state of the object using Kalman filter.
        multi_predict(stracks): Predict the next states for multiple tracks.
        multi_gmc(stracks, H): Update multiple track states using a homography matrix.
        activate(kalman_filter, frame_id): Activate a new tracklet.
        re_activate(new_track, frame_id, new_id): Reactivate a previously lost tracklet.
        update(new_track, frame_id): Update the state of a matched track.
        convert_coords(tlwh): Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah(tlwh): Convert tlwh bounding box to xyah format.
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh, score, cls):
        """Initialize new STrack instance."""
        super().__init__()
        # xywh+idx or xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

    def predict(self):
        """Predicts mean and covariance using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for given stracks."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a previously lost track with a new detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track, frame_id):
        """
        Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh):
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """Get current position in bounding box format (top left x, top left y, width, height)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self):
        """Convert bounding box to format (min x, min y, max x, max y), i.e., (top left, bottom right)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format (center x, center y, aspect ratio, height), where the aspect ratio is width /
        height.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self):
        """Get current position in bounding box format (center x, center y, width, height)."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self):
        """Get current position in bounding box format (center x, center y, width, height, angle)."""
        if self.angle is None:
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self):
        """Get current tracking results."""
        coords = self.xyxy if self.angle is None else self.xywha
        return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]

    def __repr__(self):
        """Return a string representation of the BYTETracker object with start and end frames and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"

# class STrack(BaseTrack):
#     shared_kalman = KalmanFilterXYAH()
#
#     def __init__(self, tlwh, score, feat=None, feat_history=50):
#         # wait activate
#         self._tlwh = np.asarray(tlwh, dtype=float)
#         self.kalman_filter = None
#         self.mean, self.covariance = None, None
#         self.is_activated = False
#
#         self.score = score
#         self.tracklet_len = 0
#
#         self.smooth_feat = None
#         self.curr_feat = None
#         if feat is not None:
#             self.update_features(feat)
#         self.features = deque([], maxlen=feat_history)
#         self.alpha = 0.9
#
#     def update_features(self, feat):
#         """Update features vector and smooth it using exponential moving average."""
#         feat /= np.linalg.norm(feat)
#         self.curr_feat = feat
#         if self.smooth_feat is None:
#             self.smooth_feat = feat
#         else:
#             self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
#         self.features.append(feat)
#         self.smooth_feat /= np.linalg.norm(self.smooth_feat)
#
#     def predict(self):
#         """Predicts the mean and covariance using Kalman filter."""
#         mean_state = self.mean.copy()
#         if self.state != TrackState.Tracked:
#             mean_state[6] = 0
#             mean_state[7] = 0
#
#         self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
#
#     @staticmethod
#     def multi_predict(stracks):
#         """Predicts the mean and covariance of multiple object tracks using shared Kalman filter."""
#         if len(stracks) <= 0:
#             return
#         multi_mean = np.asarray([st.mean.copy() for st in stracks])
#         multi_covariance = np.asarray([st.covariance for st in stracks])
#         for i, st in enumerate(stracks):
#             if st.state != TrackState.Tracked:
#                 multi_mean[i][6] = 0
#                 multi_mean[i][7] = 0
#         multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
#         for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
#             stracks[i].mean = mean
#             stracks[i].covariance = cov
#
#     @staticmethod
#     def multi_gmc(stracks, H=np.eye(2, 3)):
#         if len(stracks) > 0:
#             multi_mean = np.asarray([st.mean.copy() for st in stracks])
#             multi_covariance = np.asarray([st.covariance for st in stracks])
#
#             R = H[:2, :2]
#             R8x8 = np.kron(np.eye(4, dtype=float), R)
#             t = H[:2, 2]
#
#             for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
#                 mean = R8x8.dot(mean)
#                 mean[:2] += t
#                 cov = R8x8.dot(cov).dot(R8x8.transpose())
#
#                 stracks[i].mean = mean
#                 stracks[i].covariance = cov
#
#     def activate(self, kalman_filter, frame_id):
#         """Start a new tracklet"""
#         self.kalman_filter = kalman_filter
#         self.track_id = self.next_id()
#
#         self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))
#
#         self.tracklet_len = 0
#         self.state = TrackState.Tracked
#         if frame_id == 1:
#             self.is_activated = True
#         self.frame_id = frame_id
#         self.start_frame = frame_id
#
#     def re_activate(self, new_track, frame_id, new_id=False):
#         self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance)
#         if new_track.curr_feat is not None:
#             self.update_features(new_track.curr_feat)
#         self.tracklet_len = 0
#         self.state = TrackState.Tracked
#         self.is_activated = True
#         self.frame_id = frame_id
#         if new_id:
#             self.track_id = self.next_id()
#         self.score = new_track.score
#
#     def update(self, new_track, frame_id):
#         """
#         Update a matched track
#         :type new_track: STrack
#         :type frame_id: int
#         :return:
#         """
#         self.frame_id = frame_id
#         self.tracklet_len += 1
#
#         new_tlwh = new_track.tlwh
#
#         self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance)
#
#         if new_track.curr_feat is not None:
#             self.update_features(new_track.curr_feat)
#
#         self.state = TrackState.Tracked
#         self.is_activated = True
#
#         self.score = new_track.score
#
#     @property
#     def tlwh(self):
#         """Get current position in bounding box format
#         `(top left x, top left y, width, height)`.
#         """
#         if self.mean is None:
#             return self._tlwh.copy()
#         ret = self.mean[:4].copy()
#         ret[:2] -= ret[2:] / 2
#         return ret
#
#     @property
#     def tlbr(self):
#         """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
#         `(top left, bottom right)`.
#         """
#         ret = self.tlwh.copy()
#         ret[2:] += ret[:2]
#         return ret
#
#     @property
#     def xywh(self):
#         """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
#         `(top left, bottom right)`.
#         """
#         ret = self.tlwh.copy()
#         ret[:2] += ret[2:] / 2.0
#         return ret
#
#     @staticmethod
#     def tlwh_to_xyah(tlwh):
#         """Convert bounding box to format `(center x, center y, aspect ratio,
#         height)`, where the aspect ratio is `width / height`.
#         """
#         ret = np.asarray(tlwh).copy()
#         ret[:2] += ret[2:] / 2
#         ret[2] /= ret[3]
#         return ret
#
#     @staticmethod
#     def tlwh_to_xywh(tlwh):
#         """Convert bounding box to format `(center x, center y, width, height)`."""
#         ret = np.asarray(tlwh).copy()
#         ret[:2] += ret[2:] / 2
#         return ret
#
#     def to_xywh(self):
#         return self.tlwh_to_xywh(self.tlwh)
#
#     @staticmethod
#     def tlbr_to_tlwh(tlbr):
#         ret = np.asarray(tlbr).copy()
#         ret[2:] -= ret[:2]
#         return ret
#
#     @staticmethod
#     def tlwh_to_tlbr(tlwh):
#         ret = np.asarray(tlwh).copy()
#         ret[2:] += ret[:2]
#         return ret
#
#     def __repr__(self):
#         return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
#
