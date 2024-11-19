import json
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import numpy as np
from os import system, environ
from PIL import Image

from prepare_kanji import get_primitive_labels
from project import KanjiDataset


Y = get_primitive_labels()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.CenterCrop((100, 100)),
])

with open("data/kanji_data.json", "r") as file:
    data: dict = json.loads(file.read())
dataset = KanjiDataset(data, transform)


for data in dataset:
    print(data)