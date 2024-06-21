import argparse
import os
import cv2
from ultralytics import YOLO
import numpy as np

def make_parser():
    parser = argparse.ArgumentParser("Run YOLO model on Visdrone Dataset")

    parser.add_argument("--SEQUENCES_DIR",
                        type=str,
                        default=".",
                        help="Path to Visdrone dataset. It should point to train, or test directory of the dataset.")
    parser.add_argument("--WEIGHTS_PATH",
                        type=str,
                        required=True,
                        help="Path to the YOLO weight.")
    parser.add_argument("--TRACKER",
                        type=str,
                        default="bytetrack",
                        choices=["bytetrack", "botsort"],
                        help="YOLO Trackers")
    parser.add_argument("--SHOW",
                        default=False,
                        action="store_true", 
                        help="Wether to show the tracked sequences (using CV2).")
    parser.add_argument("--SAVE_RESULTS",
                        default=False,
                        action="store_true", 
                        help="save result (.txt file)")
    parser.add_argument("--SAVE_RESULTS_DIR",
                        default="./results/",
                        help="Where to save the results.txt files. Default to current ./results/")
    return parser

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
    gt_file = os.path.join(path, 'gt/gt.txt')
    det_file = os.path.join(path, 'det/det.txt')
    
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

def main(args):
    # Create window for display the result
    if args.SHOW:
        cv2.namedWindow(args.TRACKER_NAME, cv2.WINDOW_KEEPRATIO)
    
    N_SEQS = len(os.listdir(args.SEQUENCES_DIR))
    for seq_index, current_seq in enumerate(os.listdir(args.SEQUENCES_DIR)):
        model = YOLO(args.WEIGHTS_PATH)
        
        print(f'[INFO] [{seq_index+1}/{N_SEQS}] Working on {current_seq}...')
        seq_path = os.path.join(args.SEQUENCES_DIR, current_seq)
        
        # Load Groundtruth
        det, gt = parse_gt_files(seq_path)
        img1_dir = os.path.join(seq_path, 'img1')
      
        result = ''
        for tracker in model.track(img1_dir, persist=True, stream=True, tracker=f"{args.TRACKER}.yaml", verbose=False, device=0):
    
            frame_id = int(tracker.path.split('/')[-1][:-4])
            frame = tracker.orig_img
    
            frame_det = get_current_frame(gt, frame_id)
            ignored_regions = get_ignored_regions(frame_det)
         
            # Draw predictions
            for bbox in tracker.boxes:
                x, y, x2, y2 = bbox.xyxy[0]
                w, h = (x2 - x), (y2 - y)
    
                if center_in_ignored_regions(bbox.xywh[0][:2], ignored_regions):
                    pass
                else:
                    if bbox.id != None: # Only write into result if YOLO tracked this object
                        bbox_id = int(bbox.id.item())
                        result += (f'{frame_id}, {bbox_id}, {x:.4f}, {y:.4f}, {w:.4f}, {h:.4f}, {1}, -1, -1, -1\n')
                    
                    if args.SHOW:
                        cv2.rectangle(frame, 
                                      (int(x), int(y)), 
                                      (int(x) + int(w), int(y) + int(h)), 
                                      (255, 255, 0),
                                      2)

            if args.SHOW:
                cv2.imshow(args.TRACKER_NAME, frame)
                
        if args.SAVE_RESULTS:
            cv2.waitKey(1) # Can't quit the image process if result need to be save
            filename = os.path.join(args.SAVE_RESULTS_DIR, f'{current_seq}.txt')
            with open(filename, 'w') as f:
                f.write(result)
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def handle_args(args):
    """
    This function handle and process arguments so the program can run smoothly after.
    """
    args.TRACKER_NAME = os.path.basename(args.WEIGHTS_PATH)[:-3]

    # Handle sequences dir
    skipped_dir_name = ['test', 'train', 'val', '']
    args.RESULTS_DIR_NAME = []
    seq_dir_split = args.SEQUENCES_DIR.split("/")
    while seq_dir_split[-1] in skipped_dir_name:
        current_split = seq_dir_split[-1]
        seq_dir_split = seq_dir_split[:-1]
        if len(current_split) < 1:
            continue
        args.RESULTS_DIR_NAME.append(current_split)
    args.RESULTS_DIR_NAME.append(seq_dir_split[-1])
    args.RESULTS_DIR_NAME = '-'.join(args.RESULTS_DIR_NAME[::-1])

    if args.SAVE_RESULTS:
        if args.SAVE_RESULTS_DIR == './results/':
            os.makedirs(args.SAVE_RESULTS_DIR, exist_ok=True)
        args.SAVE_RESULTS_DIR = os.path.join(args.SAVE_RESULTS_DIR, args.RESULTS_DIR_NAME)
        os.makedirs(args.SAVE_RESULTS_DIR, exist_ok=True)
        
        tracker_index = 0
        while os.path.exists(os.path.join(args.SAVE_RESULTS_DIR, f'{args.TRACKER_NAME}_{tracker_index:05d}')):
            tracker_index += 1
        args.SAVE_RESULTS_DIR = os.path.join(args.SAVE_RESULTS_DIR, f'{args.TRACKER_NAME}_{tracker_index:05d}')
        os.mkdir(args.SAVE_RESULTS_DIR)
        args.SAVE_RESULTS_DIR = os.path.join(args.SAVE_RESULTS_DIR, 'data')
        os.mkdir(args.SAVE_RESULTS_DIR)        
        print(f'[INFO] Results file will be saved to {args.SAVE_RESULTS_DIR}')

    return args

if __name__ == "__main__":
    try:
        args = make_parser().parse_args()
        args = handle_args(args)
    
        main(args)
    except KeyboardInterrupt:
        print('[INFO] Stopped by User...')
        
    cv2.destroyAllWindows()
