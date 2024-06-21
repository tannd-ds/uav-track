Clone this Repo
```sh
git clone {https://github.com/tannd-ds/this_repo.git} {REPO_HOME}
```

# YOLOv8

Setup Environment, please replace {ENV_NAME} with name of your choice.
```sh
conda create -n {ENV_NAME} python=3.10
conda activate {ENV_NAME}
pip install -r requirements.txt
```

To run the demo of YOLO on Visdrone dataset (test set)
```sh
python {REPO_HOME}/my_utils/run_yolo_on_visdrone.py \
    --WEIGHTS_PATH {path_to_yolo_weights} \
    --SEQUENCES_DIR {path_to_visdrone_dataset_test/test} \
    --TRACKER botsort \
    --SAVE_RESULTS \
    --SHOW
```

# YOLOv10

> Note: YOLOv10 is not officially part of Ultralytics, we need to setup it differently.

Clone YOLOv10 Repo
```sh
git clone https://github.com/NhiNguyen34/yolov10.git {YOLOv10_HOME}
cp {REPO_HOME}/my_utils/run_yolov10_on_visdrone.py {YOLOv10_HOME}
```

> Note: You need to copy the `{REPO_HOME}/my_utils/run_yolov10_on_visdrone.py` to `{YOLOv10_HOME}` since it need to use the ultralytics of YOLOv10 instead of the official one.

Setup Environment, please replace {ENV_NAME} with name of your choice.
```sh
cd {YOLOv10_HOME}
conda create -n {ENV_NAME} python=3.10
conda activate {ENV_NAME}
pip install -r requirements.txt
```

To run the demo of YOLO on Visdrone dataset (test set)
```sh
cd {YOLOv10_HOME}
python run_yolov10_on_visdrone.py \
    --WEIGHTS_PATH {path_to_yolov10_weights} \
    --SEQUENCES_DIR {path_to_visdrone_dataset_test/test} \
    --TRACKER botsort \
    --SAVE_RESULTS \
    --SHOW
```
