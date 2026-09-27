import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

datset = torchvision.datasets.VOCDetection(root="./", year="2007", image_set="train", download=True)
datset_test = torchvision.datasets.VOCDetection(root="./", year="2007", image_set="test", download=True)

