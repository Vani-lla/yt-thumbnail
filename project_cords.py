import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from prepare_kanji import TOP_RADICALS
from project import TRANSFORM, Net

class KanjiCoordinateDataset(Dataset):
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
            kanji_entry.append(kanji.get(radical) if radical in kanji.keys() else [0, 0, 0, 0])

        return image, torch.FloatTensor(kanji_entry).flatten()
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(6, 4, (6, 6))
        self.conv3 = nn.Conv2d(4, 2, (3, 3))

        self.fc_first = nn.Linear(200, 400)
        self.fc_second = nn.Linear(400, 200)
        
        self.block = nn.ModuleDict({
            radical: nn.Sequential(
                nn.Linear(200, 140),
                nn.ReLU(),
                nn.Linear(140, 84),
                nn.ReLU(),
                nn.Linear(84, 42),
                nn.ReLU(),
                nn.Linear(42, 4),
            ) for radical in TOP_RADICALS
        })

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
        
        for conv in convs:
            x = self.pool(F.relu(conv(x.float())))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc_first(x))
        x = F.relu(self.fc_second(x))
        
        l = []
        for ind, radical in enumerate(TOP_RADICALS):
            l.append(self.block[radical](x))
        
        return torch.cat(l, dim=1)
    
with open("data/kanji_data.json", "r") as file:
    DATA: dict = json.loads(file.read())
    
    
if __name__ == "__main__":
    dataset = KanjiCoordinateDataset(DATA, TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=32//2, shuffle=True, )
    
    net = Net()
    device = torch.device('cuda:0')
    net.train()
    net.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    run = True
    epoch = 0
    while run:
        try:
            epoch += 1
            running_loss = 0.0
            
            for data in dataloader:
                optimizer.zero_grad()
                inputs, labels = data[0].to(device), data[1].to(device)
                
                outputs = net(inputs)
                loss: torch.Tensor = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            print(f'{epoch:3} loss: {running_loss:5.2f}')
        except KeyboardInterrupt:
            run = False
            
    torch.save(net, "models/model_digits.pth")
    print('Finished Training')
