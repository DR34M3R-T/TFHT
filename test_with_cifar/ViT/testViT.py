import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from ViT import *
#plt.switch_backend('agg')

img = Image.open('dog.jpg')
fig = plt.figure()
plt.imshow(img)
plt.show()

# resize to imagenet size 
transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0) # 主要是为了添加batch这个维度
x.shape

patch_size = 16 # 16 pixels
pathes = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
pathes.shape

PatchEmbedding()(x).shape

    
patches_embedded = PatchEmbedding()(x)
print("patches_embedding's shape: ", patches_embedded.shape)
MultiHeadAttention()(patches_embedded).shape

patches_embedded = PatchEmbedding()(x)
TransformerEncoderBlock()(patches_embedded).shape