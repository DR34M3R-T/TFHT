import ViT
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
#plt.switch_backend('agg')

img = Image.open('dog.jpg')
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
data1=unpickle('ViT/cifar10/data_batch_1')

test=ViT.ViT()
transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0) # 主要是为了添加batch这个维度
p=test(x)
print('\n'+str(p))
pass