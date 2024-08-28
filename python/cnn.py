import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader

device = "cpu"

transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 96, 11, 4, 0),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(96),
                nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer2 = nn.Sequential(
                nn.Conv2d(96, 256, 5, 1, 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer3 = nn.Sequential(
                nn.Conv2d(256, 384, 3, 1, 1),
                nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
                nn.Conv2d(384, 384, 3, 1, 1),
                nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
                nn.Conv2d(384, 256, 3, 1, 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(kernel_size=3, stride=2)
        )

        
        self.classifier = nn.Sequential(
                nn.Linear(256*6*6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 10)
        )


    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train(dataloader, model, loss_fn, optimizer, num_epochs):

    size = len(dataloader.dataset)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            pred = model(inputs)
            loss = loss_fn(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0




def cnn_main():

    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    
    train(trainloader, model, loss_fn, optimizer, 10)


model = CNN().to(device)
trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
for data in trainloader:
    image, label = data
    pred = model(image)
    break
