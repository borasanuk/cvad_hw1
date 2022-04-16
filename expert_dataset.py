from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json

img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])


class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""

    def __init__(self, data_root):
        self.data_root = data_root
        self.img_dir = os.path.join(self.data_root, "rgb")
        self.measurements_dir = os.path.join(self.data_root, "measurements")

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        img_path = os.path.join(self.img_dir, str(index).zfill(8) + ".png")
        img = Image.open(img_path)
        img = img_transform(img)
        measurements_path = os.path.join(
            self.measurements_dir, str(index).zfill(8) + ".json")
        with open(measurements_path, 'r') as file:
            measurements = json.load(file)
        return img, measurements
