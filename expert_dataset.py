import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np

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

    def __len__(self):
        return len([name for name in os.listdir(self.img_dir)]) - 1

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        img_path = os.path.join(self.img_dir, str(index).zfill(8) + ".png")
        img = Image.open(img_path)
        img = img_transform(img)
        measurements_path = os.path.join(
            self.measurements_dir, str(index).zfill(8) + ".json")
        with open(measurements_path, 'r') as file:
            measurements = json.load(file)

        speed = torch.tensor([measurements["speed"]], dtype=torch.float32)
        command = int(measurements["command"])
        steer = measurements["steer"]
        throttle = measurements["throttle"]
        brake = measurements["brake"]

        target_vec = np.zeros((4, 3), dtype=np.float32)
        target_vec[command, :] = np.array([steer, throttle, brake])

        mask_vec = np.zeros((4, 3), dtype=np.float32)
        mask_vec[command, :] = 1

        route_angle = measurements["route_angle"]
        lane_dist = measurements["lane_dist"]
        tl_state = measurements["tl_state"]
        tl_dist = measurements["tl_dist"]

        affordances = np.array([route_angle, lane_dist, tl_state, tl_dist])

        return img, speed, target_vec.reshape(-1), mask_vec.reshape(-1), affordances
