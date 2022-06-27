import scipy.io as sio
import numpy as np
import tkinter

#load
data = sio.loadmat('dataset/normal/normal_0.mat')
np1=np.squeeze(data['X285_DE_time'])
np2=np.squeeze(data['X285_FE_time'])
print(np1.shape)