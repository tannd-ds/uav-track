import os
import shutil
import configparser
import argparse

def make_parser():
    parser = argparse.ArgumentParser("Copy groundtruth files from VisDrone dataset (in COCO format) to creat groundtruth for TrackEval tool.")

    parser.add_argument("--BASE_DIR",
                        type=str,
                        required=True,
                        help="Relative path to VisDrone dataset directory.")
    parser.add_argument("--TARGET_DIR",
                        type=str,
                        required=True,
                        help="Relative path to the target directory for MOT17 format.")
    return parser


def visdrone_to_mot17(base_dir: str, target_dir: str, important_dirs:list=['train', 'test']):
    """
    Converts VisDrone format to MOT17 format, including 'seqinfo.ini' creation.

    Parameters:
    ----------
    base_dir: str, required
        Relative path to VisDrone dataset directory.
    target_dir: str, required
        Relative path to the target directory for MOT17 format.
    frame_rate: int, optional (default=30)
        The frame rate of the video sequences.
    im_ext: str, optional (default='.jpg')
        The file extension of the image files.
    """
    # Copy groundtruth annotations
    for d in important_dirs:
        dataset_name = os.path.basename(base_dir.rstrip(os.path.sep)) + '-' + d
        seqmaps_dir = os.path.join(target_dir, 'seqmaps')
        os.makedirs(seqmaps_dir, exist_ok=True)
        seqmaps_info = {
            'file_path': os.path.join(seqmaps_dir, f'{dataset_name}.txt'),
            'content' : 'name\n',
        }
        
        from_dir = os.path.join(base_dir, d)
        to_dir = os.path.join(target_dir, dataset_name)
        os.makedirs(to_dir, exist_ok=True)

        for seq_name in os.listdir(from_dir):
            seq_dir = os.path.join(from_dir, seq_name)
            to_seq_dir = os.path.join(to_dir, seq_name)
            os.makedirs(to_seq_dir, exist_ok=True)
            os.makedirs(os.path.join(to_dir, seq_name, 'gt'), exist_ok=True)

            ##shutil.copy(
            ##os.path.join(os.path.join(from_dir, seq_name), 'gt', 'gt.txt'), 
            ##os.path.join(os.path.join(to_dir, seq_name, 'gt'))
            ##)

            # Filter out line (detections) with class = "ignored_regions" (class==0)
            input_file = os.path.join(from_dir, seq_name, 'gt', 'gt.txt')
            output_file = os.path.join(to_dir, seq_name, 'gt', 'gt.txt')
            print(input_file, '\n', output_file)
            with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                for line in infile:
                    parts = line.strip().split(',')
                    if len(parts) >= 7 and parts[6] != '0': 
                        outfile.write(line)

            seqmaps_info['content'] += seq_name + '\n'

            # Create seqinfo.ini
            seqinfo_path = os.path.join(to_seq_dir, 'seqinfo.ini')
            config = configparser.ConfigParser()
            config['Sequence'] = {
                'name': f'{seq_name}',
                'imDir': 'img1',
                'frameRate': 10,
                'seqLength': len(os.listdir(os.path.join(seq_dir, 'img1'))),  # Count image files
                'imWidth': 1920,  # Adjust if necessary
                'imHeight': 1080, # Adjust if necessary
                'imExt': '.jpg'
            }
            with open(seqinfo_path, 'w') as f:
                config.write(f)

        # Save seqmaps
        with open(seqmaps_info['file_path'], 'w') as f:
            f.write(seqmaps_info['content'])

if __name__ == "__main__":
    args = make_parser().parse_args()
    visdrone_to_mot17(args.BASE_DIR, args.TARGET_DIR)
