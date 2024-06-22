import torch

class AssociationModel:
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def associate(self, detections):
        raise NotImplementedError
