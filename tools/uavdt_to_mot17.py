import os
import shutil


def handle_subset(subset, from_path, to_path):
    print(f'Making data for {subset} set...')
    os.mkdir(os.path.join(to_path, subset))

    for seq_name in os.listdir(os.path.join(from_path, subset, 'UAV-benchmark-M')):
        print(f'copying {seq_name}...')

        current_dir = os.path.join(to_path, subset, seq_name)

        os.mkdir(current_dir)

        # create gt, det, img1 directories
        dirs = ['gt', 'det', 'img1']
        for dir_ in dirs:
            os.mkdir(os.path.join(current_dir, dir_))

        # Copy annotation file from UAVDT to gt folder, then rename it to 'gt.txt'
        gt_original_name = f'{seq_name}_gt.txt'
        shutil.copy(
            os.path.join(from_path, subset, 'UAV-benchmark-MOTD_v1.0/GT/', gt_original_name),
            os.path.join(current_dir, 'gt')
        )

        os.rename(
            os.path.join(current_dir, 'gt', gt_original_name),
            os.path.join(current_dir, 'gt', 'gt.txt')
        )

        # Copy annotation file from visdrone to gt folder, then rename it to 'det.txt'
        # This is currently use gt to "simulate" det
        # TODO: Find better solution.
        shutil.copy(
            os.path.join(from_path, subset, 'UAV-benchmark-MOTD_v1.0/GT/', gt_original_name),
            os.path.join(current_dir, 'det')
        )

        os.rename(
            os.path.join(current_dir, 'det', gt_original_name),
            os.path.join(current_dir, 'det', 'det.txt')
        )

        seq_dir = os.path.join(from_path, subset, 'UAV-benchmark-M', seq_name)
        for frame in os.listdir(seq_dir):
            shutil.copy(
                os.path.join(seq_dir, frame),
                os.path.join(current_dir, 'img1')
            )


def visdrone_to_mot17(visdrone_path: str):
    """
    parameters:
    ----------
    visdrone_path: str, required
        relative path to visdrone dataset directory.
    """
    parent_dir = os.path.dirname(visdrone_path)
    visdrone_coco_dir = os.path.join(parent_dir, os.path.basename(visdrone_path) + '_coco')

    # Create root directory
    try:
        os.mkdir(visdrone_coco_dir)
    except FileExistsError:
        print(f'Stopped. Make sure folder {visdrone_coco_dir} directory is empty.')
        return

    handle_subset(subset='train', from_path=visdrone_path, to_path=visdrone_coco_dir)
    handle_subset(subset='test', from_path=visdrone_path, to_path=visdrone_coco_dir)


if __name__ == '__main__':
    visdrone_to_mot17('../datasets/UAVDT')
