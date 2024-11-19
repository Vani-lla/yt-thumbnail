import torch
import numpy as np
from project import *
from PIL import Image


device = torch.device('cuda:0')

net: Net = torch.load("models/model.pth", weights_only=False)
net.to(device)
net.eval()

accuracy = {k: [] for k in TOP_RADICALS}
img = Image.open("data/test/4.png").convert("RGB")
img = TRANSFORM(img)
img = img.reshape((1, 1, 100, 100))

img = img.to(device)
result = net(img)

for i, radical in enumerate(TOP_RADICALS):
    r = result[0][i].item()
    print(radical, round(r, 2))


# print(net)
