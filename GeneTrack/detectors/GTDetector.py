import os

import numpy as np

from . import DetectorBase, Detection


class GTDetector(DetectorBase):
    def __init__(self, groundtruth_path: str, first_frame_index: int = 1):
        super().__init__(None)
        self.det = None
        self.gt = None
        self.parse_gt_files(groundtruth_path)
        self.current_frame = first_frame_index

        self.current_ignored_regions = []

    def detect(self, img):
        dets_list = self.get_current_frame()
        self.current_ignored_regions = self.get_ignored_regions(dets_list)

        dets_np = np.array(dets_list)
        xywh = dets_np[:, 2:6]
        # Calculate left_x and top_y
        half_width = xywh[:, 2] / 2
        half_height = xywh[:, 3] / 2
        left_x = xywh[:, 0] + half_width
        top_y = xywh[:, 1] + half_height
        # Create the new boxes array (left_x, top_y, width, height)
        new_boxes = np.column_stack((left_x, top_y, xywh[:, 2], xywh[:, 3]))

        cls = dets_np[:, 6]
        conf = np.array([1 for _ in dets_list])

        dets = Detection(conf=conf,
                         xywh=new_boxes,
                         cls=cls)

        self.current_frame += 1
        return dets

    def parse_gt_files(self, path):
        """
        Parses ground truth (GT) and detection (DET) files from a specified directory path.

        This function assumes the GT and DET files follow a specific CSV format (see "Data Format").
        It reads the files, processes the data, and returns parsed lists of GT and DET entries.

        Parameters
        ----------
        path : str
            The path to the directory containing the 'gt/gt.txt' and 'det/det.txt' files.

        Returns
        -------
        det_parsed : list
            A list of lists, where each inner list represents a detection entry with 10 float values:
            [frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z].
        gt_parsed : list
            A list of lists, where each inner list represents a ground truth entry with the same 10
            float values as the detection entries.

        Data Format
        ----------
        The tracker file format should be the same as the ground truth file, which is a CSV text-file containing one object instance per line.
        Each line must contain 10 values:
            <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        The world coordinates x,y,z are ignored for the 2D challenge and can be filled with -1.
        Similarly, the bounding boxes are ignored for the 3D challenge. However, each line is still required to contain 10 values.
        """
        gt_file = os.path.realpath(os.path.join(os.getcwd(), path, 'gt/gt.txt'))
        det_file = os.path.realpath(os.path.join(os.getcwd(), path, 'det/det.txt'))

        with open(gt_file, 'r') as gt_file, open(det_file, 'r') as det_file:
            gt = gt_file.readlines()
            det = det_file.readlines()

        gt = [line.replace('\n', '').split(',') for line in gt]
        gt_parsed = []
        for line in gt:
            gt_parsed.append([float(num) for num in line])

        det = [line.replace('\n', '').split(',') for line in det]
        det_parsed = []
        for line in det:
            det_parsed.append([float(num) for num in line])

        self.gt = gt_parsed
        self.det = det_parsed

    def get_current_frame(self):
        """
        Filters an annotation data list to return only lines corresponding to a specific frame.

        Returns
        -------
        list
            A new list containing only the annotation lines from `anno_data` where the first
            element (presumably the frame ID) matches `frame_id`.
        """
        dets = [line for line in self.gt if line[0] == self.current_frame]
        return dets

    def get_ignored_regions(self, dets, ignored_id=0):
        """
        Extracts ignored regions from a list of detections.

        Parameters
        ----------
        dets : list
            A list of detection data, where each element is a list representing a detected object.
            Assumes that the 7th element (index 6) in each detection list indicates whether the
            region is ignored (0) or not (1).
        ignored_id: int
            ID of the ignored_regions.

        Returns
        -------
        list
            A new list containing only the detection data for objects marked as ignored.
        """
        return [d for d in dets if d[6] == ignored_id]

    def filter_objects_in_ignored_regions(self, dets: list):
        """"""
        filtered = []
        for det in dets:
            x, y = det[0] + det[2] / 2, det[1] + det[3] / 2
            for region in self.current_ignored_regions:
                greater_than_tl = x > region[2] and y > region[3]
                smaller_than_br = x < (region[2] + region[4]) and y < (region[3] + region[5])
                if not (greater_than_tl and smaller_than_br):
                    filtered.append(det)
        return filtered
