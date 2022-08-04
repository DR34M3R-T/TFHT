from cProfile import label
from mattool import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['font.family'] = ['Times New Roman']
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markersize'] = 4
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

info = loadmat('./result/CWRU/p&d10/mat/FC0-IN0-bs64-ps16-d128-dp6-h6-dk64-md256.mat')

epoch_num = len(info['repeat'][0]['epochs'])
train_loss = []
train_loss_x = [i/5.0+0.2 for i in range(epoch_num*5)]
train_loss_e = []
train_loss_e_x = [i+1 for i in range(epoch_num)]
test_loss = []
test_loss_x = [i+1 for i in range(epoch_num)]
test_acc = []
test_acc_x = [i+1 for i in range(epoch_num)]

for epoch in info['repeat'][0]['epochs']:
    train_loss.extend(epoch['train_loss'][0:5])
    train_loss_e.append(epoch['train_loss'][4])
    test_loss.append(epoch['test_loss'])
    test_acc.append(epoch['test_acc'])
    pass

fig, plts = plt.subplots(1,3)
plts[0].plot(train_loss_x, train_loss,'ro-',markersize=1.5)
plts[0].plot(train_loss_e_x, train_loss_e,'ro')
plts[0].plot([],[],'ro-',label='Training set')
plts[0].set_xlabel('Epochs')
plts[0].set_ylabel('Loss')
plts[0].set_ylim(bottom=0)
plts[0].legend()

plts[1].plot(test_loss_x, test_loss,'b^-',label='Test set')
plts[1].set_xlabel('Epochs')
plts[1].set_ylabel('Loss')
plts[1].set_ylim(bottom=0)
plts[1].legend()

plts[2].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda i,p: '%1.0f'%(100*i)+'%'))
plts[2].plot(test_acc_x, test_acc,'b^-',label='Test set')
plts[2].set_xlabel('Epochs')
plts[2].set_ylabel('Acc.')
plts[2].set_ylim(top=1)
plts[2].legend()

plt.suptitle('ViT Model')

fig.show()

pass