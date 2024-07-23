import numpy as np

class Detection(object):
    """Basic and General output format for GeneTrack detectors.
    IMPORTANT: x and y in xywh is the coordinates of the center of the detector.
    """

    def __init__(self, conf: np.ndarray, xywh: np.ndarray, cls: np.ndarray):
        self.conf = conf
        self.xywh = xywh
        self.cls = cls

    @property
    def xywhr(self) -> np.ndarray:
        return self.xywh

    @property
    def xyxy(self) -> np.ndarray:
        return_value = self.xywh.copy()
        return_value[:, 0] = return_value[:, 0] - return_value[:, 2] / 2
        return_value[:, 1] = return_value[:, 1] - return_value[:, 3] / 2
        return_value[:, 2] = return_value[:, 0] + return_value[:, 2]
        return_value[:, 3] = return_value[:, 1] + return_value[:, 3]
        return return_value

    @staticmethod
    def join_detections(detections):
        if len(detections) == 0:
            return detections

        new_conf = np.concatenate([det.conf for det in detections], axis=0)
        new_xywh = np.concatenate([det.xywh for det in detections], axis=0)
        new_cls = np.concatenate([det.cls for det in detections], axis=0)

        return Detection(new_conf, new_xywh, new_cls)

    def __copy__(self):
        new_detection = Detection(self.conf, self.xywh, self.cls)
        return new_detection
