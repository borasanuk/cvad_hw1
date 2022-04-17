import torch
import torch.nn as nn
from torchvision.models import resnet18


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self):
        super(AffordancePredictor, self).__init__()

        self.perception = nn.Sequential(
            resnet18(pretrained=True),
            nn.Linear(1000, 4),
        )

    def forward(self, img):
        output = self.perception(img)

        return output
