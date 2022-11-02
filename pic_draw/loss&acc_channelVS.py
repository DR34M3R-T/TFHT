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
mpl.rcParams['boxplot.notch'] = False
mpl.rcParams['boxplot.whiskerprops.linestyle'] = '--'
mpl.rcParams['boxplot.flierprops.marker'] = 'x'
mpl.rcParams['boxplot.flierprops.markersize'] = 2.5

# 导入.mat文件
info = loadmat('./result/CWRU/p&d10/channelVS/FC0-IN0-bs32-ps128-d64-dp4-h4-dk64-md128.mat')
info_FC = loadmat('./result/CWRU/p&d10/channelVS/FC1-IN1-bs32-ps128-d64-dp4-h4-dk64-md128.mat')

repeat_num = len(info['repeat'])
epoch_num = len(info['repeat'][0]['epochs'])
train_loss_epochs = np.zeros((epoch_num,repeat_num,5))
train_loss_epochs_x = [i/5.0+0.2 for i in range(epoch_num*5)]

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
        train_loss_epochs[epoch,repeat]=k['train_loss_list'][0:5]
        train_loss[epoch,repeat]=k['train_loss']
        train_acc[epoch,repeat]=k['train_acc']
        test_loss[epoch,repeat]=k['test_loss']
        test_acc[epoch,repeat]=k['test_acc']
        pass

repeat_num = len(info_FC['repeat'])
epoch_num = len(info_FC['repeat'][0]['epochs'])
train_loss_epochs_FC = np.zeros((epoch_num,repeat_num,5))
train_loss_epochs_x_FC = [i/5.0+0.2 for i in range(epoch_num*5)]

train_loss_FC = np.zeros((epoch_num,repeat_num))
train_loss_x_FC = [i+1 for i in range(epoch_num)]
train_acc_FC = np.zeros((epoch_num,repeat_num))
train_acc_x_FC = [i+1 for i in range(epoch_num)]
test_loss_FC = np.zeros((epoch_num,repeat_num))
test_loss_x_FC = [i+1 for i in range(epoch_num)]
test_acc_FC = np.zeros((epoch_num,repeat_num))
test_acc_x_FC = [i+1 for i in range(epoch_num)]
for repeat in range(repeat_num):
    for epoch in range(epoch_num):
        k = info_FC['repeat'][repeat]['epochs'][epoch]
        train_loss_epochs_FC[epoch,repeat]=k['train_loss_list'][0:5]
        train_loss_FC[epoch,repeat]=k['train_loss']
        train_acc_FC[epoch,repeat]=k['train_acc']
        test_loss_FC[epoch,repeat]=k['test_loss']
        test_acc_FC[epoch,repeat]=k['test_acc']
        pass

fig, plts = plt.subplots(2,2)
# plts[0][0].plot(train_loss_epochs_x[0:10], train_loss_epochs[0:10],'ro-',markersize=1.5)
# plts[0][0].plot([],[],'ro-',label='Training set')
plts[0][0].boxplot(np.transpose(train_loss[0:10]))
plts[0][0].plot(test_loss_x[0:10], np.median(test_loss[0:10],axis=1), 'b^-',label='1 Channel')
plts[0][0].boxplot(np.transpose(train_loss_FC[0:10]))
plts[0][0].plot(test_loss_x_FC[0:10], np.median(test_loss_FC[0:10],axis=1), 'ro-',label='3 Channel')
plts[0][0].set_xlabel('Epochs')
plts[0][0].set_ylabel('Loss')
plts[0][0].set_ylim(top=0.1,bottom=0)
plts[0][0].legend(loc='upper right')

plts[0][1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda i,p: '%1.0f'%(100*i)+'%'))
plts[0][1].boxplot(np.transpose(test_acc[0:10]))
plts[0][1].plot(test_acc_x[0:10] , np.median(test_acc[0:10],axis=1) , 'b^-',label='1 Channel')
plts[0][1].boxplot(np.transpose(test_acc_FC[0:10]))
plts[0][1].plot(test_acc_x_FC[0:10] , np.median(test_acc_FC[0:10],axis=1) , 'ro-',label='3 Channel')
plts[0][1].set_xlabel('Epochs')
plts[0][1].set_ylabel('Acc.')
plts[0][1].set_ylim(top=1,bottom=0.95)
plts[0][1].legend(loc='lower right')

#plts[0][2].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda i,p: '%1.0f'%(100*i)+'%'))
plts[1][0].plot(test_loss_x , np.mean(test_loss,axis=1) , 'b^-', markersize=2.5, label='1 Channel')
plts[1][0].plot(test_loss_x_FC, np.mean(test_loss_FC,axis=1), 'ro-', markersize=2.5, label='3 Channel')
plts[1][0].set_xlabel('Epochs')
plts[1][0].set_ylabel('Average Loss')
plts[1][0].set_ylim(top=0.05,bottom=0)
plts[1][0].legend(loc='upper right')

plts[1][1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda i,p: '%1.1f'%(100*i)+'%'))
plts[1][1].plot(test_acc_x , np.mean(test_acc,axis=1) , 'b^-', markersize=2.5, label='1 Channel')
plts[1][1].plot(test_acc_x_FC, np.mean(test_acc_FC,axis=1), 'ro-', markersize=2.5, label='3 Channel')
plts[1][1].set_xlabel('Epochs')
plts[1][1].set_ylabel('Average Acc.')
plts[1][1].set_ylim(top=1,bottom=0.99)
plts[1][1].legend(loc='lower right')

'''
plts[1][0].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda i,p: '%1.0f'%(100*i)+'%'))
plts[1][0].boxplot(np.transpose(train_acc[0:10]))
plts[1][0].plot(train_acc_x[0:10], np.median(train_acc[0:10],axis=1), 'ro-',label='Training set')
plts[1][0].set_xlabel('Epochs')
plts[1][0].set_ylabel('Acc.')
plts[1][0].set_ylim(top=1,bottom=0.5)
plts[1][0].legend(loc='lower right')

plts[1][1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda i,p: '%1.0f'%(100*i)+'%'))
plts[1][1].boxplot(np.transpose(test_acc[0:10]))
plts[1][1].plot(test_acc_x[0:10] , np.median(test_acc[0:10],axis=1) , 'b^-',label='Test set')
plts[1][1].set_xlabel('Epochs')
plts[1][1].set_ylabel('Acc.')
plts[1][1].set_ylim(top=1,bottom=0.5)
plts[1][1].legend(loc='lower right')

plts[1][2].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda i,p: '%1.0f'%(100*i)+'%'))
plts[1][2].plot(test_acc_x , np.mean(test_acc,axis=1) , 'b^-', markersize=2.5, label='Test set')
plts[1][2].plot(train_acc_x, np.mean(train_acc,axis=1), 'ro-', markersize=2.5, label='Training set')
plts[1][2].set_xlabel('Epochs')
plts[1][2].set_ylabel('Average Acc.')
plts[1][2].set_ylim(top=1,bottom=0.5)
plts[1][2].legend(loc='lower right')
'''

plt.suptitle('ViT Model')

fig.show()

pass