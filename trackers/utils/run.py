import os
from argparse import Namespace

import cv2

from . import visdrone


def run(model, seq_path, args):
    if isinstance(args, dict):
        current_seq: str = seq_path.split('/')[-1]
        tracker: str = f"{args.get('TRACKER', 'botsort')}.yaml"
        tracker_name: str = args.get('TRACKER_name', 'botsort')
        show_video: bool = args.get('SHOW', False)
        save_results: bool = args.get('SAVE_RESULTS', False)
        save_results_dir: str = args.get('SAVE_RESULTS_DIR', os.getcwd())
    elif isinstance(args, Namespace):
        current_seq: str = seq_path.split('/')[-1]
        tracker: str = f"{args.TRACKER}.yaml"
        tracker_name: str = args.TRACKER_NAME
        show_video: bool = args.SHOW
        save_results: bool = args.SAVE_RESULTS
        save_results_dir: str = args.SAVE_RESULTS_DIR

    # Load Groundtruth
    det, gt = visdrone.parse_gt_files(seq_path)
    img1_dir = os.path.join(seq_path, 'img1')

    result = ''
    for img_file in sorted(os.listdir(img1_dir)):
        frame_id = int(img_file[:-4])
        frame = cv2.imread(os.path.join(img1_dir, img_file))

        frame_det = visdrone.get_current_frame(gt, frame_id)
        ignored_regions = visdrone.get_ignored_regions(frame_det)

        dets = model.track(frame, frame_id=frame_id, persist=True, tracker=tracker, verbose=False, device=0)
        for det in dets:
            if det.track_id > 0 and not visdrone.center_in_ignored_regions(
                    (det.bb_left + det.bb_width / 2, det.bb_top + det.bb_height / 2),
                    ignored_regions):
                cv2.rectangle(
                    frame,
                    (int(det.bb_left), int(det.bb_top)),
                    (int(det.bb_left + det.bb_width),
                     int(det.bb_top + det.bb_height)),
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    frame,
                    str(det.track_id),
                    (int(det.bb_left), int(det.bb_top)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                result += f'{frame_id}, {det.track_id}, {det.bb_left:.4f}, {det.bb_top:.4f}, {det.bb_width:.4f}, {det.bb_height:.4f}, {0.8}, -1, -1, -1\n'

        if show_video:
            cv2.imshow(tracker_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if save_results:
                    print('[INFO] Can\'t skip the Video when --SAVE_RESULTS==True...')
                else:
                    break

        if save_results:
            filename = os.path.join(save_results_dir, f'{current_seq}.txt')
            with open(filename, 'w') as f:
                f.write(result)
