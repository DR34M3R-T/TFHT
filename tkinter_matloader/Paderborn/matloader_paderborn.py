import scipy.io as sio
import numpy as np
import tkinter

#load
data = sio.loadmat('tkinter_matloader/N09_M07_F10_K001_1.mat')
tata = data['N09_M07_F10_K001_1']['Y'][0][0][0][5][1]
#np1=np.squeeze(data['X285_DE_time'])
#np2=np.squeeze(data['X285_FE_time'])
print("np1.shape")