import torch
from torch import nn
from vit_pytorch import Transformer
from einops import repeat
from einops.layers.torch import Rearrange

class CNN1D(nn.Module):
    def __init__(self, *, num_classes,path_num=2,channels=1):
        super(CNN1D, self).__init__()
        self.path_num = path_num
        self.flatten = nn.ModuleList(
            [nn.Flatten() for i in range(self.path_num)]
        )
        self.conv = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=128, stride=16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(16, 32, kernel_size=3, stride=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(32, 64, kernel_size=3, stride=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(64, 128, kernel_size=3, stride=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(128, 256, kernel_size=3, stride=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
            ) for i in range(self.path_num)]
        )

        self.to_latent = nn.Sequential(
            nn.Linear(256*path_num, 100*path_num),
            nn.ReLU()
        )

        self.out = nn.Linear(100*path_num, num_classes)

    def forward(self, sig, feature_out=False):
        x=['','']
        for i in range(self.path_num):
            #x[i] = sig[:,i].unsqueeze(1)
            x[i] = self.conv[i](sig)
            x[i] = self.flatten[i](x[i])
        x_total = torch.cat(tuple(x[k] for k in range(self.path_num)),dim=1)
        features = self.to_latent(x_total)

        if feature_out: return x_total
        return self.out(features)