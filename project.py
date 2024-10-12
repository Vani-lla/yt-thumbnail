import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from os import system

system("export HSA_OVERRIDE_GFX_VERSION=11.0.0")

# Helper imports

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)

transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    # transforms.RandomAutocontrast(p=1.0),
    # transforms.RandomPosterize(bits=2, p=1.0)
    transforms.GaussianBlur((5, 9), sigma=(.2, 5)),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
])


dataset = datasets.ImageFolder("data/thumbnails", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, )
testloader = DataLoader(dataset, batch_size=32, shuffle=False, )

dataiter = iter(testloader)
images, labels = next(dataiter)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


for i in range(5):
    imshow(images[i])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)  # 12x475x635
        self.pool = nn.MaxPool2d(5, 5)  # 12x95x127
        self.conv2 = nn.Conv2d(12, 24, 5)  # 24*18*24
        self.fc1 = nn.Linear(24*18*24, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x.float())))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
device = torch.device('cuda:0')
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
        running_loss = 0.0

torch.save(net.state_dict(), "models/model.pt")
print('Finished Training')

classes = ("failure", "succesfull")

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
print(total_pred, correct_pred)
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
