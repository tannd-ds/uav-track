import argparse
import os
import cv2
from ultralytics import YOLO
import trackers
from trackers.YOLOModel import YOLOModel
from trackers import utils

def make_parser():
    parser = argparse.ArgumentParser("Run YOLO model on VisDrone Dataset")

    parser.add_argument("--SEQUENCES_DIR",
                        type=str,
                        default=".",
                        help="Path to VisDrone dataset. It should point to train, or test directory of the dataset.")
    parser.add_argument("--MODEL",
                        type=str,
                        required=True,
                        help="The model used for prediction.")
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
                        help="Whether to show the tracked sequences (using CV2).")
    parser.add_argument("--SAVE_RESULTS",
                        default=False,
                        action="store_true",
                        help="save result (.txt file)")
    parser.add_argument("--SAVE_RESULTS_DIR",
                        default="./results/",
                        help="Where to save the results.txt files. Default to current ./results/")
    return parser

def create_model(args):
    """Create model based on user choice."""
    if args.MODEL == "yolo":
        return YOLOModel(weights_path=args.WEIGHTS_PATH)
    else:
        raise ValueError("Invalid model type")

def main(args):
    # Create window for display the result
    if args.SHOW:
        cv2.namedWindow(args.TRACKER_NAME, cv2.WINDOW_KEEPRATIO)
    
    N_SEQS = len(os.listdir(args.SEQUENCES_DIR))
    for seq_index, current_seq in enumerate(os.listdir(args.SEQUENCES_DIR)):
        # model = YOLO(args.WEIGHTS_PATH)
        model = create_model(args)

        print(f'[INFO] [{seq_index+1}/{N_SEQS}] Working on {current_seq}...')
        seq_path = os.path.join(args.SEQUENCES_DIR, current_seq)
        
        # Load Groundtruth
        det, gt = utils.visdrone.parse_gt_files(seq_path)
        img1_dir = os.path.join(seq_path, 'img1')
      
        result = ''
        for tracker in model.track(img1_dir, persist=True, stream=True, tracker=f"{args.TRACKER}.yaml", verbose=False, device=0):
    
            frame_id = int(tracker.path.split('/')[-1][:-4])
            frame = tracker.orig_img
    
            frame_det = utils.visdrone.get_current_frame(gt, frame_id)
            ignored_regions = utils.visdrone.get_ignored_regions(frame_det)
         
            # Draw predictions
            for bbox in tracker.boxes:
                x, y, x2, y2 = bbox.xyxy[0]
                w, h = (x2 - x), (y2 - y)
    
                if utils.visdrone.center_in_ignored_regions(bbox.xywh[0][:2], ignored_regions):
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
                if cv2.waitKey(1) & 0xFF == ord('q') and not args.SAVE_RESULTS:
                    break
                
        if args.SAVE_RESULTS:
            filename = os.path.join(args.SAVE_RESULTS_DIR, f'{current_seq}.txt')
            with open(filename, 'w') as f:
                f.write(result)


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
