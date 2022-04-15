from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json


class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""

    def __init__(self, data_root):
        self.data_root = data_root
        self.img_dir = os.path.join(self.data_root, "rgb")
        self.measurements_dir = os.path.join(self.data_root, "measurements")

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        img_path = os.path.join(self.img_dir, str(index).zfill(8) + ".png")
        img = read_image(img_path)
        measurements_path = os.path.join(
            self.measurements_dir, str(index).zfill(8) + ".json")
        with open(measurements_path, 'r') as file:
            measurements = json.load(file)
        return img, measurements
