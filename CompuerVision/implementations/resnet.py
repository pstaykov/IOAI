import torch
import torch.nn as nn
from torchvision.transforms import v2
import torchvision
from torch.utils.data import DataLoader

transforms = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=3),
    v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=3),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

training_data = torchvision.datasets.FashionMNIST(root="./", download=True, train=True, transform=transforms)
test_data = torchvision.datasets.FashionMNIST(root="./", download=True, train=False, transform=test_transforms)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

resnet = torchvision.models.resnet18(weights = "DEFAULT")

num_features = resnet.fc.in_features

resnet.fc = nn.Identity()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet18 = resnet
        self.fc1 = nn.Linear(in_features=num_features, out_features=10)

    def forward(self, x):
        return self.fc1(self.resnet18(x))

model = Net()

model.train()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr = 1e-3)

for _ in range(10):
    for x,y in train_dataloader:
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()