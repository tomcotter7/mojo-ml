import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cpu"

training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
)

test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
)


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.mlp(x)

def train(dataloader, model, loss_fn, optimizer, num_epochs):
    size = len(dataloader.dataset)
    model.train()
    
    for _ in range(num_epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 500 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    
def mlp_main():
    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_dataloader = DataLoader(training_data, batch_size=64)

    train(train_dataloader, model, loss_fn, optimizer, 5)

    test_dataloader = DataLoader(test_data, batch_size=64)

    model.eval()
    size = len(test_dataloader.dataset) # type: ignore
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


shape = (2, 3, 4)
x = torch.rand(shape)
print(x[:, 0, :])
