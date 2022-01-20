import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from vit_pytorch import ViT
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
learning_rate = 1e-3
batch_size = 64
epochs = 5

train_data=datasets.CIFAR10(
    root="dataset\\cifar",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data=datasets.CIFAR10(
    root="dataset\\cifar",
    train=False,
    download=True,
    transform=ToTensor()
)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#ViT

v = ViT(
    image_size = 32,
    patch_size = 8,
    num_classes = 10,
    dim = 128,
    depth = 4,
    heads = 12,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)

# Initialize the loss function
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(v.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, v, loss_fn, optimizer)
    test_loop(test_dataloader, v, loss_fn)
print("Done!")