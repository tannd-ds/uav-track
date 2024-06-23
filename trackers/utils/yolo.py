from ultralytics.engine.results import Results

from trackers.detection.base import Detection


def yolo_results_to_base_detection(results: list[Results]):
    """Converts YOLOv8 detection results into a list of `Detection` objects.

    This function processes the raw output from a YOLOv8 model and transforms it into a format
    compatible with tracking systems that expect `Detection` objects.

    Parameters:
    ----------
        results (Results): A `Results` object from the Ultralytics YOLOv8 library,
            containing information about detected objects in an image.

    Returns:
    ----------

        list[Detection]: A list of `Detection` objects, each representing a single
            detected object with attributes like bounding box coordinates, confidence score,
            and class ID.

    Raises:
    ----------
        TypeError: If the input `results` is not an instance of `Results`.

    Example:
    ----------

        ```
        from ultralytics import YOLO

        # ... (load YOLO model and perform inference)

        results = model(image)  # Get YOLO results
        detections = yolo_results_to_base_detection(results)
        ```
    """
    dets = []
    for box in results[0].boxes:
        conf = box.conf.cpu().numpy()[0]
        bbox = box.xyxy.cpu().numpy()[0]
        cls_id = box.cls.cpu().numpy()[0]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        det = Detection(int(box.id.item()))
        det.bb_left = bbox[0]
        det.bb_top = bbox[1]
        det.bb_width = w
        det.bb_height = h
        det.conf = conf
        det.det_class = cls_id
        det.track_id = int(box.id.item())

        dets.append(det)
    return dets
