from cProfile import label
from mattool import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 图像默认设置
mpl.rcParams['font.family'] = ['Times New Roman']
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markersize'] = 4
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['boxplot.notch'] = True
mpl.rcParams['boxplot.whiskerprops.linestyle'] = '--'
mpl.rcParams['boxplot.boxprops.linewidth'] = 0.5
mpl.rcParams['boxplot.medianprops.linewidth'] = 0.5
mpl.rcParams['boxplot.boxprops.linewidth'] = 0.5
mpl.rcParams['boxplot.whiskerprops.linewidth'] = 0.5
mpl.rcParams['boxplot.flierprops.marker'] = 'x'
mpl.rcParams['boxplot.flierprops.markersize'] = 2.5

# 导入.mat文件
info = loadmat('./result/CWRU/p&d10/mat/FC0-IN0-bs64-ps256-d32-dp1-h2-dk64-md64.mat')

repeat_num = len(info['repeat'])
epoch_num = len(info['repeat'][0]['epochs'])
train_loss_epochs = np.zeros((epoch_num*5,repeat_num))
train_loss_epochs_x = [(i/5.0+0.2) for i in range(epoch_num*5)]

train_loss = np.zeros((epoch_num,repeat_num))
train_loss_x = [i+1 for i in range(epoch_num)]
train_acc = np.zeros((epoch_num,repeat_num))
train_acc_x = [i+1 for i in range(epoch_num)]
test_loss = np.zeros((epoch_num,repeat_num))
test_loss_x = [i+1 for i in range(epoch_num)]
test_acc = np.zeros((epoch_num,repeat_num))
test_acc_x = [i+1 for i in range(epoch_num)]
for repeat in range(repeat_num):
    for epoch in range(epoch_num):
        k = info['repeat'][repeat]['epochs'][epoch]
        train_loss_epochs[epoch*5:(epoch+1)*5,repeat]=k['train_loss_list'][0:5]
        train_loss[epoch,repeat]=k['train_loss']
        train_acc[epoch,repeat]=k['train_acc']
        test_loss[epoch,repeat]=k['test_loss']
        test_acc[epoch,repeat]=k['test_acc']
        pass

fig, plts = plt.subplots()
plts.boxplot(np.transpose(train_loss_epochs[0:50]),positions=train_loss_epochs_x[0:50], widths=0.1)
plts.plot(train_loss_x[0:10],  np.mean(train_loss[0:10],axis=1),'ro-')
plts.plot(train_loss_epochs_x[0:50], np.median(train_loss_epochs[0:50],axis=1), 'g^-', label='During training', markersize=2, linewidth=0.75)
plts.plot([],[],'ro-',label='After training')
plts.set_xticks(np.arange(0, 11, 1), range(11))
plts.set_xlabel('Epochs')
plts.set_ylabel('Loss')
plts.set_ylim(bottom=0)
plts.legend()

plt.suptitle('Loss Proformance During and After Every Epoch of Training')

fig.show()

pass