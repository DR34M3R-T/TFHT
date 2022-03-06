import scipy.io as sio
import numpy as np

#load
data = sio.loadmat('dataset/12k_fan_end/B007_3.mat')
np1=np.squeeze(data['X097_DE_time'])
np2=np.squeeze(data['X097_FE_time'])
print(np1.shape)