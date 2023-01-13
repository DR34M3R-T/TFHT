import torch
from torch import nn
from vit_pytorch import Transformer
from einops import repeat
from einops.layers.torch import Rearrange

#ViT
class ViT(nn.Module):
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

    def forward(self, img, feature_out=False):
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
        x_total = torch.cat(tuple(x[k] for k in range(self.path_num)),dim=1)
        x_total = self.to_latent(x_total)
        if feature_out: return x_total
        return self.mlp_head(x_total)