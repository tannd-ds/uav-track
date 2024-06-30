import numpy as np


class Detection(object):
    """Basic and General output format for GeneTrack detectors."""

    def __init__(self, conf: np.ndarray, xywh: np.ndarray, cls: np.ndarray):
        self.conf = conf
        self.xywh = xywh
        self.cls = cls

    @property
    def xywhr(self) -> np.ndarray:
        return self.xywh
