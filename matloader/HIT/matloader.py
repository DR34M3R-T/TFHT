import scipy.io as sio
import numpy as np
from random import shuffle

files = [
    ['inner10.mat',2],
    ['health.mat',0],
    ['health2.mat',0],
    ['inner05.mat',1],
    ['outer05.mat',1],
]

nplabel = None
npdata = None
npspeed = None
npwidth = None
flag = None

for matname, w in files:
    mat = sio.loadmat('./dataset/HIT/origin/'+matname)
    mat = mat[matname[:-4]]
    data = mat[:,0:6,:]
    speed = mat[:,6,0:2]
    label = int(mat[0,7,0])
    data_cutted = np.vstack(np.moveaxis(np.split(data,10,axis=2),0,1))
    label = np.zeros(data_cutted.shape[0],dtype='int8')+label
    width = np.zeros(data_cutted.shape[0],dtype='int8')+w
    speed_repeated = np.repeat(speed,10,axis=0)

    if flag == None:
        nplabel = label
        npdata = data_cutted
        npspeed = speed_repeated
        npwidth = width
        flag = 1
    else:
        nplabel = np.hstack((nplabel,label))
        npdata = np.vstack((npdata,data_cutted))
        npspeed = np.vstack((npspeed,speed_repeated))
        npwidth = np.hstack((npwidth,width))
    #if w==2: test = np.unique(data[0:,0:6],return_index=True)
    pass
pass

index = [i for i in range(len(npdata))]
shuffle(index)

pass

nplabel = nplabel[index]
npdata = npdata[index]
npspeed = npspeed[index]
npwidth = npwidth[index]

np.save('./dataset/HIT/data.npy',npdata)
np.save('./dataset/HIT/label.npy',nplabel)
np.save('./dataset/HIT/speed.npy',npspeed)
np.save('./dataset/HIT/width.npy',npwidth)
pass