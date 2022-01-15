from imageio import imsave
import numpy as np
import pickle as pkl
 
 
def unpickle(file):
    fo = open(file, 'rb')
    dict = pkl.load(fo,encoding='bytes')
    fo.close()
    return dict
 
 
for j in range(1, 6):
    dataName = "ViT\cifar10\data_batch_" + str(j)
    Xtr = unpickle(dataName)
    print (dataName + " is loading...")
 
    for i in range(0, 10000):
        img = np.reshape(Xtr[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        picName = 'train/' + str(Xtr[b'labels'][i]) + '/' + str(i + (j - 1)*10000) + '.jpg'
        import os
        if not os.path.exists('train/' + str(Xtr[b'labels'][i]) + '/'):
            os.mkdir('/train/' + str(Xtr[b'labels'][i]) + '/')
        imsave(picName, img)
    print(dataName + " loaded.")
 
print ("test_batch is loading...")
 
 
testXtr = unpickle("test_batch")
for i in range(0, 10000):
    img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'test/' + str(testXtr[b'labels'][i]) + '_' + str(i) + '.jpg'
    imsave(picName, img)
print ("test_batch loaded.")