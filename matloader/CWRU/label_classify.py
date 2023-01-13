import numpy as np

label = np.load('dataset/CWRU_lapped/relabel.npy')
'''
no_FE = label[label[:,6]==1]
no_FE = no_FE[:,6]+\
    10*no_FE[:,5]+\
    100*no_FE[:,4]+\
    100000*no_FE[:,3]+\
    1000000*no_FE[:,2]+\
    100000000*no_FE[:,1]+\
    1000000000*no_FE[:,0]
no_FE = np.unique(no_FE)

no_BA = label[label[:,5]==1]
no_BA = no_BA[:,6]+\
    10*no_BA[:,5]+\
    100*no_BA[:,4]+\
    100000*no_BA[:,3]+\
    1000000*no_BA[:,2]+\
    100000000*no_BA[:,1]+\
    1000000000*no_BA[:,0]
no_BA = np.unique(no_BA)'''

#故障位置 4类
relabel = np.zeros(label.shape[0],np.int32)-1
relabel[label[:,0]==0]=0 #normal
relabel[label[:,1]==1]=1 #B
relabel[label[:,1]==2]=2 #IR
relabel[label[:,1]==3]=3 #OR
relabel[np.logical_and(label[:,2]==28,label[:,1]==1)]=101 #DE: B28
relabel[np.logical_and(label[:,2]==28,label[:,1]==2)]=102 #DE: IR28
np.save('dataset/CWRU_lapped/p4.npy',relabel)

#故障端+故障位置 7类
relabel = np.zeros(label.shape[0],np.int32)-1
relabel[label[:,0]==0]=0 #normal
relabel[np.logical_and(label[:,0]==1,label[:,1]==1)]=1 #DE: B IR OR
relabel[np.logical_and(label[:,0]==1,label[:,1]==2)]=2 
relabel[np.logical_and(label[:,0]==1,label[:,1]==3)]=3
relabel[np.logical_and(label[:,0]==2,label[:,1]==1)]=4 #FE: B IR OR
relabel[np.logical_and(label[:,0]==2,label[:,1]==2)]=5
relabel[np.logical_and(label[:,0]==2,label[:,1]==3)]=6
relabel[np.logical_and(label[:,2]==28,label[:,1]==1)]=101 #DE: B28
relabel[np.logical_and(label[:,2]==28,label[:,1]==2)]=102 #DE: IR28
np.save('dataset/CWRU_lapped/e&p7.npy',relabel)

#故障端+故障直径 7类
relabel = np.zeros(label.shape[0],np.int32)-1
relabel[label[:,0]==0]=0 #normal
relabel[np.logical_and(label[:,0]==1,label[:,2]==7)]=1 #DE: 7 14 21
relabel[np.logical_and(label[:,0]==1,label[:,2]==14)]=2 
relabel[np.logical_and(label[:,0]==1,label[:,2]==21)]=3
relabel[np.logical_and(label[:,0]==2,label[:,2]==7)]=4 #FE: 7 14 21
relabel[np.logical_and(label[:,0]==2,label[:,2]==14)]=5
relabel[np.logical_and(label[:,0]==2,label[:,2]==21)]=6
np.save('dataset/CWRU_lapped/e&d7.npy',relabel)

#故障位置+故障直径 10类 
relabel = np.zeros(label.shape[0],np.int32)-1
relabel[label[:,0]==0]=0 #normal
relabel[np.logical_and(label[:,1]==1,label[:,2]==7)]=1 #B: 7 14 21
relabel[np.logical_and(label[:,1]==1,label[:,2]==14)]=2 
relabel[np.logical_and(label[:,1]==1,label[:,2]==21)]=3
relabel[np.logical_and(label[:,1]==2,label[:,2]==7)]=4 #IR: 7 14 21
relabel[np.logical_and(label[:,1]==2,label[:,2]==14)]=5
relabel[np.logical_and(label[:,1]==2,label[:,2]==21)]=6
relabel[np.logical_and(label[:,1]==3,label[:,2]==7)]=7 #OR: 7 14 21
relabel[np.logical_and(label[:,1]==3,label[:,2]==14)]=8
relabel[np.logical_and(label[:,1]==3,label[:,2]==21)]=9
np.save('dataset/CWRU_lapped/p&d10.npy',relabel)

pass