import json
import torch
from matplotlib import pyplot as plt
from project import *
import numpy as np


device = torch.device('cuda:0')

net: Net = torch.load("models/model.pth", weights_only=False)
net.to(device)
net.eval()

dataset = KanjiDataset(DATA, TRANSFORM)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, )

accuracy = {k: [] for k in TOP_RADICALS}

for data in dataloader:
    inputs, labels = data[0].to(device), data[1].to(device)

    result = net(inputs)
    for i, radical in enumerate(TOP_RADICALS):
        l, r = labels[0][i].item(), result[0][i].item()
        if l == 1:
            if r >= .6:
                accuracy[radical].append(1)
            else:
                accuracy[radical].append(0)
        else:
            if r < .6:
                accuracy[radical].append(1)
            else:
                accuracy[radical].append(0)
                
for key, val in accuracy.items():
    print(f"{key}: {np.average(val)}")

# print(net)
