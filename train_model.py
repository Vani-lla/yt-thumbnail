import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from project_cords import KanjiCoordinateDataset, Net
from prepare_kanji import TOP_RADICALS, RADICAL_WEIGHTS
from project import TRANSFORM

with open("data/kanji_data.json", "r") as file:
    DATA: dict = json.loads(file.read())
    
if __name__ == "__main__":
    dataset = KanjiCoordinateDataset(DATA, TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, )
    
    net: Net = torch.load("models/model_digits_prime.pth", weights_only=False)
    device = torch.device('cuda:0')
    net.train()
    net.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005)
    
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