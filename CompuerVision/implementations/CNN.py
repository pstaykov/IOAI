import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

dataset = torchvision.datasets.CIFAR10(root = "./", train = True, download=True, transform=transforms)
dataset_test = torchvision.datasets.CIFAR10(root = "./", train = False, download=True, transform=transforms)

dataloader_train = torch.utils.data.DataLoader(dataset=dataset, shuffle = True, batch_size=64)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, shuffle=False, batch_size=64)

def conv(i,o):
    return(nn.Sequential(
        nn.Conv2d(i,o,3),
        nn.ReLU()
        ))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = conv(3,16) # 32 -> 30
        self.maxpool = nn.MaxPool2d(2) # 30 -> 15
        self.conv2 = conv(16,32) # 15 -> 13
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(13*13*32, 10) # 10 objects

    def forward(self,x):
        return(self.fc(self.flatten(self.conv2(self.maxpool(self.conv1(x))))))

model = Net()

optimizer = torch.optim.AdamW(params=model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

for _ in range(10):
    for x, y in (dataloader_train):
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

