import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from vit_pytorch import ViT,Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
learning_rate = 1e-3
batch_size = 64
epochs = 5
x_train = torch.from_numpy(np.load('./dataset/XJTU/xTrain.npy'))
y_train = torch.from_numpy(np.load('./dataset/XJTU/yTrain.npy'))
x_test = torch.from_numpy(np.load('./dataset/XJTU/xTest.npy'))
y_test = torch.from_numpy(np.load('./dataset/XJTU/yTest.npy'))


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 自定义dataset和数据预处理
preprocess = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=0, std=1)
])

class BearFaultDataset(Dataset):
    def __init__(self, inputs, targets, transform, reshape):
        if reshape:
            self.inputs = inputs[:, :2025].reshape((-1, 45, 45))
        else:
            self.inputs = inputs
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        input = self.inputs[item]
        target = self.targets[item]
        # input = self.transform(input)
        return input, target


# 实例化dataset
isreshape = False
training_data = BearFaultDataset(x_train, y_train, transform=preprocess, reshape=isreshape)
test_data = BearFaultDataset(x_test, y_test, transform=preprocess, reshape=isreshape)
print(training_data.inputs.shape, test_data.inputs.shape)
# 定义dataloader
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#ViT



class myViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('(h p1) (w p2) -> (h w) (p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        print(img)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

v = myViT(
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