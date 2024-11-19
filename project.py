import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

from prepare_kanji import TOP_RADICALS

# Detect GPU
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)
# export HSA_OVERRIDE_GFX_VERSION=11.0.0


class KanjiDataset(Dataset):
    def __init__(self, data: dict[str, dict[str, list[float]]], transform):
        self.data = data
        self.keys = list(data.keys())
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        kanji_id = self.keys[index]

        kanji = self.data[kanji_id]

        image = Image.open(f"data/kanji/{kanji_id}.png").convert("RGB")
        image = TRANSFORM(image)

        kanji_entry: list[int] = []
        for radical in TOP_RADICALS:
            kanji_entry.append(1 if radical in kanji.keys() else 0)

        return image, torch.FloatTensor(kanji_entry)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(6, 4, (6, 6))
        self.conv3 = nn.Conv2d(4, 2, (3, 3))

        self.fc1 = nn.Linear(200, 140)
        self.fc2 = nn.Linear(140, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 42)
        self.fc5 = nn.Linear(42, len(TOP_RADICALS))

        self.pool = nn.MaxPool2d((2, 2), (2, 2))  # (H - 2)/2 + 1 = H/2

    def forward(self, x):
        convs = [
            self.conv1,  # 100x100
            # 6x50x50 after pooling
            self.conv2,  # 45x45
            # 4x22x22 after pooling
            self.conv3  # 19x19
            # 2x10x10 after pooling
        ]
        fcs = [
            self.fc1,
            self.fc2,
            self.fc3,
            self.fc4,
            self.fc5,
        ]
        for conv in convs:
            x = self.pool(F.relu(conv(x.float())))

        x = torch.flatten(x, 1)
        for fc in fcs[:-1]:
            x = F.relu(fc(x))

        x = F.sigmoid(fcs[-1](x))

        return x


TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize((106, 106)),
    transforms.CenterCrop((100, 100)),
])

with open("data/kanji_data.json", "r") as file:
    DATA: dict = json.loads(file.read())

if __name__ == "__main__":
    dataset = KanjiDataset(DATA, TRANSFORM)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [
                                                                0.8, 0.2])
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, )

    net = Net()
    device = torch.device('cuda:0')
    net.train()
    net.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(50):
        running_loss = 0.0

        for data in dataloader:
            optimizer.zero_grad()

            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        accuracy = {k: [] for k in TOP_RADICALS}
        for data in test_dataloader:
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

        print(f'{epoch + 1:3} loss: {running_loss:5.2f}')
        for key, val in accuracy.items():
            print(f"{key}: {np.average(val)}")

    torch.save(net, "models/model.pth")
    print('Finished Training')
