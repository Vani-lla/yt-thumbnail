import torch
import numpy as np
from project_cords import *
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

matplotlib.rcParams['font.family'] = 'HanaMinA'

device = torch.device('cuda:0')

net: Net = torch.load("models/model_digits_beta.pth", weights_only=False)
net.to(device)
net.eval()
    
with open("data/kanji_data.json", "r") as file:
    DATA: dict = json.loads(file.read())
    
# for kanji_id in DATA.keys():
kanji_id = "0732b"
accuracy = {k: [] for k in TOP_RADICALS}
orig = Image.open(f"data/kanji/{kanji_id}.png").convert("RGB")
img = TRANSFORM(orig)
img = img.reshape((1, 1, 100, 100))

img = img.to(device)
result:torch.Tensor = net(img)

fig, ax = plt.subplots()
ax.imshow(orig, cmap="gray")

for ind, result in enumerate(np.split(result.cpu().detach().numpy()[0, :], len(TOP_RADICALS))):
    print(TOP_RADICALS[ind], np.round(result, 1))
    
    if all(result > 2):
        rect = patches.Rectangle((result[0], result[1]), result[2]-result[0], result[3]-result[1], linewidth=2, edgecolor='r', facecolor='r', alpha=.6)
        ax.text(result[0], result[1]-3, TOP_RADICALS[ind], color='green', fontsize=24, ha='center', va='center')
        ax.add_patch(rect)

for radical in DATA[kanji_id]:
    result = DATA[kanji_id][radical]
    ax.add_patch(patches.Rectangle((result[0], result[1]), result[2]-result[0], result[3]-result[1], linewidth=2, edgecolor='b', facecolor='b', alpha=.4))

        
# torch.onnx.export(net, img, "model.onnx", input_names=["Input Image"], output_names=["Bounding Boxes"])

plt.show()