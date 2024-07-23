import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from . import Detection


class DetectorBase:
    def __init__(self, model, verbose=False):
        self.model = model
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For turning `numpy` array to `torch` array
        self.preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect(self, image) -> Detection:
        raise NotImplementedError

    def numpy_to_tensor(self, image, preprocessor=None):
        """Turn `numpy` image to `torch` image"""
        assert isinstance(image, np.ndarray), "Input image must be of type 'np.ndarray'"
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        if preprocessor is not None:
            image = preprocessor(image)
        else:
            image = self.preprocessor(image)
        return image.to(self.device)

    def __call__(self, image):
        return self.detect(image)
