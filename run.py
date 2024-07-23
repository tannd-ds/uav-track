import argparse
import os

import cv2
import yaml

from GeneTrack import run
from GeneTrack.detectors import YOLODetector, FasterRCNNDetector
from GeneTrack.trackers import BYTETrack, BOTSORT, SORT


class SupportedMethods(object):
    detectors = ["yolo", "fasterrcnn", "pro"]
    trackers = ["sort", "bytetrack", "botsort"]

def make_parser():
    parser = argparse.ArgumentParser("Run YOLO model on VisDrone Dataset")

    parser.add_argument("--SEQUENCES_DIR",
                        type=str,
                        default=".",
                        help="Path to VisDrone dataset. It should point to train, or test directory of the dataset.")
    parser.add_argument("--DETECTOR",
                        type=str,
                        choices=SupportedMethods.detectors,
                        help="The model used for prediction.")
    parser.add_argument("--WEIGHTS_PATH",
                        type=str,
                        required=True,
                        help="Path to the YOLO weight.")
    parser.add_argument("--TRACKER",
                        type=str,
                        default="bytetrack",
                        choices=SupportedMethods.trackers,
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


def parse_tracker_config(args):
    cfg_path = os.path.join('GeneTrack/trackers/configs/', f'{args.TRACKER}.yaml')
    with open(cfg_path, 'r') as f:
        tracker_config = yaml.safe_load(f)

    # Add config values to args
    for key, value in tracker_config.items():
        if hasattr(args, key):
            continue
        else:
            setattr(args, key, value)

    return tracker_config

def create_model(args):
    """Create model based on user choice."""
    detector = None
    if args.DETECTOR == "yolo":
        detector = YOLODetector(weight_path=args.WEIGHTS_PATH)
    elif args.DETECTOR == "fasterrcnn":
        detector = FasterRCNNDetector(weights_path=args.WEIGHTS_PATH)
    else:
        raise ValueError(f"Invalid detector type, expected {SupportedMethods.detectors} but got '{args.DETECTOR}'")

    tracker = None
    if args.TRACKER == "bytetrack":
        tracker = BYTETrack(args)
    elif args.TRACKER == "botsort":
        tracker = BOTSORT(args)
    elif args.TRACKER == "sort":
        tracker = SORT(args)
    else:
        raise ValueError(f"Invalid tracker type, expected {SupportedMethods.trackers} but got '{args.TRACKER}'")

    return detector, tracker


def main(args):
    # Create window for display the result
    if args.SHOW:
        cv2.namedWindow(args.TRACKER_NAME, cv2.WINDOW_KEEPRATIO)

    n_seqs = len(os.listdir(args.SEQUENCES_DIR))
    """
    IMPORTANT: For simplicity and fast inference, for UAVDT dataset, we only infer in 10 first sequences of the dataset.
    """
    for seq_index, current_seq in enumerate(sorted(os.listdir(args.SEQUENCES_DIR))):
        print(f'[INFO] [{seq_index + 1}/{n_seqs}] Working on {current_seq}...')
        detector, tracker = create_model(args)

        seq_path = os.path.join(args.SEQUENCES_DIR, current_seq)
        run.run(detector, tracker, seq_path, args)


def handle_args(args):
    """
    This function handle and process arguments so the program can run smoothly after.
    """
    parse_tracker_config(args)

    args.TRACKER_NAME = f'{args.DETECTOR}_{args.TRACKER}_{os.path.basename(args.WEIGHTS_PATH)[:-3]}'

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

    # Save args to yaml
    if args.SAVE_RESULTS:
        with open(os.path.join(args.SAVE_RESULTS_DIR, 'args.yaml'), 'w') as f:
            yaml.dump(vars(args), f)

    return args


if __name__ == "__main__":
    try:
        args = make_parser().parse_args()
        args = handle_args(args)

        main(args)
    except KeyboardInterrupt:
        print('[INFO] Stopped by User...')

    if args.SHOW:
        cv2.destroyAllWindows()
