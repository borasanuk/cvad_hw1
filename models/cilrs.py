import torch
import torch.nn as nn
from torchvision.models import resnet18


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""

    def __init__(self):
        super(CILRS, self).__init__()

        self.perception = nn.Sequential(
            resnet18(pretrained=True),
            nn.Linear(1000, 512),
        )

        self.speed_fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.Linear(128, 128),
        )

        self.emb_fc = nn.Sequential(
            nn.Linear(512+128, 512),
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Linear(256, 256),
                nn.Linear(256, 3),
            ) for i in range(4)
        ])  # a branch for each high-level command

        self.speed_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 1),
        )

    def forward(self, img, speed):

        img = self.perception(img)
        speed = self.speed_fc(speed)

        emb = torch.cat([img, speed], dim=1)
        emb = self.emb_fc(emb)

        output = torch.cat([branch(emb) for branch in self.branches], dim=1)

        pred_speed = self.speed_branch(img)

        return output, pred_speed
