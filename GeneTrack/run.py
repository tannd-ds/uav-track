import os
import re
import time
from argparse import Namespace

import cv2

from . import utils
from . import visdrone
from .detectors import GTDetector


def run(detector, tracker, seq_path, args):
    if isinstance(args, Namespace):
        args = vars(args)
    current_seq: str = seq_path.split('/')[-1]
    tracker_name: str = args.get('TRACKER_NAME', 'botsort')
    show_video: bool = args.get('SHOW', False)
    save_results: bool = args.get('SAVE_RESULTS', False)
    save_results_dir: str = args.get('SAVE_RESULTS_DIR', os.getcwd())

    # Load Groundtruth
    gt_detector = GTDetector(groundtruth_path=seq_path)
    img1_dir = os.path.join(seq_path, 'img1')

    run_time = 0
    n_image = len(os.listdir(img1_dir))

    result = ''
    for img_file in sorted(os.listdir(img1_dir)):
        frame_id = int(''.join(re.findall(pattern='\d+', string=img_file[:-4])))
        frame = cv2.imread(os.path.join(img1_dir, img_file))
        frame = cv2.resize(frame, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)

        gt_det = gt_detector.detect(frame)

        start = time.time()
        dets = detector(frame)

        tracklets = tracker.update(dets)
        run_time += time.time() - start
        for tracklet in tracklets:
            xyxy = tracklet[0], tracklet[1], tracklet[2], tracklet[3]
            track_id = str(int(tracklet[4]))

            if visdrone.center_in_ignored_regions(
                    ((tracklet[0] + tracklet[2]) / 2,
                     (tracklet[1] + tracklet[3]) / 2),
                    gt_detector.current_ignored_regions):
                continue

            utils.plot_one_box(xyxy,
                               frame,
                               color=utils.colors[int(tracklet[6]) % len(utils.colors)],
                               label=track_id,
                               line_thickness=2)

            result += f'{frame_id}, {track_id}, {tracklet[0]:.4f}, {tracklet[1]:.4f}, {(tracklet[2] - tracklet[0]):.4f}, {(tracklet[3] - tracklet[1]):.4f}, {1}, -1, -1, -1\n'

        if show_video:
            cv2.imshow(tracker_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if save_results:
                    print('[INFO] Can\'t skip the Video when --SAVE_RESULTS==True...')
                else:
                    break

    print(f"[INFO] Avg run time: {run_time / n_image:.2f}")

    if save_results:
        filename = os.path.join(save_results_dir, f'{current_seq}.txt')
        with open(filename, 'w') as f:
            f.write(result)
