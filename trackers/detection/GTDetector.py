import os

from trackers.utils.visdrone import get_current_frame, parse_gt_files
from .base import DetectionModel


class GTDetector(DetectionModel):
    """This detector return Groundtruth detections."""
    def __init__(self, gt_dir):
        super().__init__()

        assert gt_dir is not None, "GTDetector requires a ground truth file"
        self.gt_dir = gt_dir
        self.current_anno_file = {
            "filename": "",
            "data": [],
        }

    def update_anno_file(self, new_name):
        self.current_anno_file["filename"] = new_name
        self.current_anno_file["data"], _ = parse_gt_files(
            os.path.join(self.gt_dir, self.current_anno_file["filename"])
        )

    def detect(self, image, **kwargs):
        frame_id = kwargs.get("frame_id", -1)
        sequence_name = kwargs.get("sequence_name", "")
        assert frame_id != -1, "GTDetector requires a frame id"
        assert sequence_name != "", "GTDetector requires a sequence name"

        if self.current_anno_file["filename"] == "" or self.current_anno_file["data"] == []:
            self.update_anno_file(sequence_name)
        return get_current_frame(self.current_anno_file["data"], frame_id)