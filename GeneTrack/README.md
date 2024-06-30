# Getting started with GenTrack - A Highly Customizable MOT Pipeline

## Want to implement a new Detector?

- **Step 1**: Create your Detector `.py` inside `GeneTrack/detectors/`, for example I'll create
  a `CustomYOLODetector.py`, with following content:

```python
from ultralytics import YOLO
from . import DetectorBase, Detection


class CustomYOLODetector(DetectorBase):
    def __init__(self, weight_path: str):
        model = YOLO(weight_path)
        super().__init__(model=model)

    def detect(self, img) -> Detection:
        yolo_dets = self.model(img, verbose=self.verbose, device=self.device)[0].cpu().numpy()
        dets = Detection(yolo_dets.boxes.conf, yolo_dets.boxes.xywh, yolo_dets.boxes.cls)
        return dets

    def __call__(self, img):
        return self.detect(img)

```

### Explanation:

```python
from . import DetectorBase, Detection
```

Import important classes: You have to inherit from `DetectorBase` and your custom Detector need to return a `Detection`
object.

```python
class CustomYOLODetector(DetectorBase):
    pass
```

The `CustomYOLODetector` class inherits from the `DetectorBase` class. This is important for consistency within the
GenTrack framework. The `DetectorBase` class provides a standardized interface for detectors, ensuring that all
detectors in the pipeline adhere to the same input/output format and basic functionality.

```python
def __init__(self, weight_path: str):
    pass
```

This constructor takes the path to a YOLO weight file (weight_path) as input.
It initializes the underlying YOLO model from the provided weight file.
The `super().__init__(model=model)` call initializes the parent class (`DetectorBase`) with the loaded YOLO model.

```python
def detect(self, img) -> Detection:
    pass
```

This method is the core of the object detection process. It takes an image (img) as input.
**IMPORTANT**: What ever types of output you got, you have to turn it into `Detection` object.

**Step 2**: Now after successfully creating your custom `Detector` class, you can simply call it in your program:

```python
from GeneTrack.detectors.CustomYOLODetector import CustomYOLODetector

my_detector = CustomYOLODetector('/path/to/my/ultimate/yolo/model.pt')
detections = my_detector(image)
```