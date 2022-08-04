import torch
from torch import nn
from vit_pytorch import Transformer
from einops import repeat
from einops.layers.torch import Rearrange

#ViT
class MyCNN1D(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', path_num = 2, channels=3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0 , 'Image dimensions must be divisible by the patch size.'
        num_patches = image_size // patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        patch_dim = patch_size * channels
        self.path_num = path_num
        
        self.to_patch_embedding = nn.ModuleList([
            nn.Sequential(
                Rearrange('b c (l p)-> b l (p c)',p=patch_size),
                nn.Linear(patch_dim, dim),
            ) for i in range(self.path_num)])

        self.pos_embedding = nn.ParameterList([
            nn.Parameter(torch.randn(1, num_patches + 1, dim))
            for i in range(self.path_num)])
        self.cls_token = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, dim))
            for i in range(self.path_num)])
        self.dropout = nn.ModuleList([
            nn.Dropout(emb_dropout)
            for i in range(self.path_num)])

        self.transformer = nn.ModuleList([
            Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
            for i in range(self.path_num)])

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim*path_num),
            nn.Linear(dim*path_num, num_classes)
        )

    def forward(self, img):
        #print(img.size())
        x=['',''];
        for i in range(self.path_num):
            x[i] = self.to_patch_embedding[i](img[:,i])
            b, n, _ = x[i].shape

            cls_tokens = repeat(self.cls_token[i], '() n d -> b n d', b = b)
            x[i] = torch.cat((cls_tokens, x[i]), dim=1)
            x[i] += self.pos_embedding[i][:, :(n + 1)]
            x[i] = self.dropout[i](x[i])

            x[i] = self.transformer[i](x[i])

            x[i] = x[i].mean(dim = 1) if self.pool == 'mean' else x[i][:, 0]
        x_total = torch.cat((x[0],x[1]),dim=1)
        x_total = self.to_latent(x_total)
        x_total = self.mlp_head(x_total)
        return x_total

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

    def forward(self, sig):
        x=['','']
        for i in range(self.path_num):
            #x[i] = sig[:,i].unsqueeze(1)
            x[i] = self.conv[i](sig[:,i])
            x[i] = self.flatten[i](x[i])
        x_total = torch.cat(tuple(x[k] for k in range(self.path_num)),dim=1)
        features = self.to_latent(x_total)
        return self.out(features)