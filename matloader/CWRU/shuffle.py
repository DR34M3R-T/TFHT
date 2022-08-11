from random import shuffle
import numpy as np

label = np.load('dataset/CWRU/label.npy')
data = np.load('dataset/CWRU/data.npy')


index = [i for i in range(len(data))]
shuffle(index)

label = label[index]
data = data[index]

np.save('dataset/CWRU/label.npy',label)
np.save('dataset/CWRU/data.npy',data)