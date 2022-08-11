from mattool import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

info_list = []
paraname = 'ps'
para_list = []

ps=8
while ps<=2048:
    file_name = f'./result/CWRU/p&d10/mat/FC0-IN0-bs32-ps{ps}-d128-dp6-h6-dk64-md256.mat'
    info = loadmat(file_name)
    repeat_num = len(info['repeat'])
    epoch_num = len(info['repeat'][0]['epochs'])

    train_loss = np.zeros((epoch_num,repeat_num))
    train_acc = np.zeros((epoch_num,repeat_num))
    test_loss = np.zeros((epoch_num,repeat_num))
    test_acc = np.zeros((epoch_num,repeat_num))

    for repeat in range(repeat_num):
        for epoch in range(epoch_num):
            k = info['repeat'][repeat]['epochs'][epoch]
            train_loss[epoch,repeat]=k['train_loss']
            train_acc[epoch,repeat]=k['train_acc']
            test_loss[epoch,repeat]=k['test_loss']
            test_acc[epoch,repeat]=k['test_acc']
            pass
    info_list.append([train_loss,train_acc,test_loss,test_acc])
    para_list.append(ps)
    ps *= 2

info_list = np.array(info_list)
epoch = 7
fig, ax = plt.subplots()
# ax[0][0].plot(train_loss_epochs_x[0:10], train_loss_epochs[0:10],'ro-',markersize=1.5)
# ax[0][0].plot([],[],'ro-',label='Training set')
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda i,p: '%1.2f'%(100*i)+'%'))
train_acc = info_list[:,1,epoch]
ax.plot(np.arange(9),np.mean(train_acc,axis=1), 'ro-')
ax.boxplot(np.transpose(train_acc),positions=np.arange(9))
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda i,p: f'{para_list[int(i)]}'))
ax.set_xlabel(paraname)
ax.set_ylabel('Acc.')
ax.set_ylim(top=1)
ax.set_title(f'para. {paraname}\'s influence on Acc. at epoch {epoch+1}')
plt.show()

pass