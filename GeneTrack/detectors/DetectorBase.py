import torch


class DetectorBase:
    def __init__(self, model, verbose=False):
        self.model = model
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def detect(self, image):
        raise NotImplementedError
