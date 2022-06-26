import torch
from torch import nn
from vit_pytorch import Transformer
from einops import repeat
from einops.layers.torch import Rearrange
x=1
#ViT
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, 
                 dim, pool = 'cls', channels = 2):
        super().__init__()
        assert image_size % patch_size == 0, \
              'Image dimensions must be divisible by the patch size.'
        num_patches = image_size // patch_size
        assert pool in {'cls', 'mean'}, \
              'pool type must be either cls (cls token) or mean (mean pooling)'
        patch_dim = channels * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (l p)-> b l (p c)',p=patch_size),
            nn.Linear(patch_dim, dim)
        )
        #other init steps...

    def forward(self, img):
        #forward method...
        return self.mlp_head(x)

def pair(t):
    pass

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size,
                 dim, pool = 'cls', channels = 3):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert  image_height % patch_height == 0 \
            and image_width % patch_width == 0, \
               'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
                      (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert(pool in {'cls', 'mean'},
              'pool type must be either cls (cls token) or mean (mean pooling)')

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1 = patch_height, 
                p2 = patch_width),
            nn.Linear(patch_dim, dim)
        )
        #other init steps...

    def forward(self, img):
        #forward method...
        return self.mlp_head(x)

print('you can \
    write like  this.')
