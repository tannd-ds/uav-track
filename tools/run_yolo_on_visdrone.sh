python tools/run_yolo_on_visdrone.py \
    --WEIGHTS_PATH ../uav/yolov8l_visdrone.pt \
    --SEQUENCES_DIR ../datasets/VisDrone2019-MOT_coco/train/ \
    --TRACKER botsort \
    --SAVE_RESULTS \
    --SHOW
