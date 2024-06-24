import argparse
import os

import cv2

import trackers
from trackers import AssembleModel, YOLOModel, utils


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
                        default="results",
                        help="Where to save the results.txt files. Default to results/")
    return parser

def create_model(args):
    """Create model based on user choice."""
    if args.MODEL == "yolo":
        return YOLOModel(weights_path=args.WEIGHTS_PATH)
    elif args.MODEL == "ucmc":
        return AssembleModel(
            detector=trackers.detection.UCMCDetector('demo/ucmc/cam_para.txt', args.WEIGHTS_PATH),
            associator=trackers.association.UCMCAssociator()
        )
    else:
        raise ValueError("Unsupported model type, got {}".format(args.MODEL))

def main(args):
    # Create window for display the result
    if args.SHOW:
        cv2.namedWindow(args.TRACKER_NAME, cv2.WINDOW_KEEPRATIO)
    
    n_seqs = len(os.listdir(args.SEQUENCES_DIR))
    for seq_index, current_seq in enumerate(os.listdir(args.SEQUENCES_DIR)):
        model = create_model(args)

        print(f'[INFO] [{seq_index+1}/{n_seqs}] Working on {current_seq}...')
        seq_path = os.path.join(args.SEQUENCES_DIR, current_seq)
        utils.run(model, seq_path, args)

def handle_args(args):
    """
    This function handle and process arguments so the program can run smoothly after.
    """
    args.TRACKER_NAME = os.path.basename(args.WEIGHTS_PATH)[:-3]

    # Handle sequences dir
    skipped_dir_name = ['test', 'train', 'val']
    args.RESULTS_DIR_NAME = []
    seq_dir_split = args.SEQUENCES_DIR.rstrip("/").split("/")
    while seq_dir_split[-1] in skipped_dir_name:
        current_split = seq_dir_split[-1]
        seq_dir_split = seq_dir_split[:-1]
        if len(current_split) < 1:
            continue
        args.RESULTS_DIR_NAME.append(current_split)
    args.RESULTS_DIR_NAME.append(seq_dir_split[-1])
    args.RESULTS_DIR_NAME = '-'.join(args.RESULTS_DIR_NAME[::-1])

    if args.SAVE_RESULTS:
        if args.SAVE_RESULTS_DIR == 'results':
            if os.path.exists('TrackEval'):
                args.SAVE_RESULTS_DIR = 'TrackEval/'
                dirs_list = 'data/trackers/mot_challenge'.split('/')
                for curr_dir in dirs_list:
                    args.SAVE_RESULTS_DIR = os.path.join(args.SAVE_RESULTS_DIR, curr_dir)
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
