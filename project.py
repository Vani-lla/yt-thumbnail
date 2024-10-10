import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np

# Helper imports
import matplotlib.pyplot as plt

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)

transform = transforms.Compose([
    # transforms.Resize((480, 640)),
    transforms.ToTensor(),
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

dataset = datasets.ImageFolder("data/thumbnails", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

images, labels = next(iter(dataloader))
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    print(images[ii])