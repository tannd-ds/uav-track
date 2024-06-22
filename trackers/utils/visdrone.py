import os

def parse_gt_files(path):
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

    with open(gt_file, 'r') as file:
        gt = file.readlines()

    gt = [line.replace('\n', '').split(',') for line in gt]
    gt_parsed = []
    for line in gt:
        gt_parsed.append([float(num) for num in line])

    with open(det_file, 'r') as file:
        det = file.readlines()

    det = [line.replace('\n', '').split(',') for line in det]
    det_parsed = []
    for line in det:
        det_parsed.append([float(num) for num in line])

    return det_parsed, gt_parsed

def get_current_frame(anno_data, frame_id):
    """
    Filters an annotation data list to return only lines corresponding to a specific frame.

    Parameters
    ----------
    anno_data : list
        A list of annotation data, where each element is a list containing information about
        a detected object in a video frame (e.g., frame ID, object ID, coordinates, etc.).
    frame_id : int
        The ID of the frame for which to retrieve annotation lines.

    Returns
    -------
    list
        A new list containing only the annotation lines from `anno_data` where the first
        element (presumably the frame ID) matches `frame_id`.
    """
    return [line for line in anno_data if line[0] == frame_id]

def get_ignored_regions(dets, ignored_id=0):
    """
    Extracts ignored regions from a list of detections.

    Parameters
    ----------
    dets : list
        A list of detection data, where each element is a list representing a detected object.
        Assumes that the 7th element (index 6) in each detection list indicates whether the
        region is ignored (0) or not (1).
    ignored_id: int
        Id of the ignored_regions.

    Returns
    -------
    list
        A new list containing only the detection data for objects marked as ignored.
    """
    return [d for d in dets if d[6] == ignored_id]

def center_in_ignored_regions(center, ignored_regions):
    """
    Checks if a given center point lies within any of the specified ignored regions.

    Parameters
    ----------
    center : tuple (int, int)
        The (x, y) coordinates of the center point.
    ignored_regions : list
        A list of ignored regions, where each region is represented as a list with the following
        format: [frame_id, obj_id, x_top_left, y_top_left, width, height].

    Returns
    -------
    bool
        True if the center point falls within any of the ignored regions, False otherwise.
    """
    x, y = center
    for region in ignored_regions:
        greater_than_tl = x > region[2] and y > region[3]
        smaller_than_br = x < (region[2] + region[4]) and y < (region[3] + region[5])
        if greater_than_tl and smaller_than_br:
            return True
    return False