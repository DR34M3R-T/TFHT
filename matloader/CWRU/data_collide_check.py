'''检查数据集是否有重复'''
import numpy as np
import scipy.io as sio

data = np.load('./dataset/CWRU/data.npy') #7253 3 2048
label = np.load('./dataset/CWRU/label.npy') #7253 7

t = np.unique(data,axis=0,return_inverse=True,return_index=True,return_counts=True)

data_collide = data[t[1][t[3]==2]]
label_collide = label[t[1][t[3]==2]]
label_collide_values = np.unique(label_collide)

pass
