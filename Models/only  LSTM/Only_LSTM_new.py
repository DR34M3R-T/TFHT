import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
import os
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import Only_LSTM_Function
#os.chdir(r'D:/VScode/python-vit/Rotor-fault-vit/dataset/TSNE/')
# 设定训练用的设备
device = "cuda" if torch.cuda.is_available() else "cpu"
# 打印看一下
print("Using {} device".format(device))

FullChannel=False
IgnoreNormal=False

# 导入raw数据集
label_name='p&d10.npy'
data = torch.from_numpy(np.load('D:/VScode/python-vit/Rotor-fault-vit/dataset/CWRU/data.npy')) #7253 3 2048
print(data.size())
label = torch.from_numpy(np.load('D:/VScode/python-vit/Rotor-fault-vit/dataset/CWRU/'+label_name)) #7253
label = label.type(torch.LongTensor)
print("In train:{}.".format(label_name))

print("Full Channel" if FullChannel else "One Channel")
#print(data.shape)
data=data[label!=-1]
#print(data.shape)
label=label[label!=-1]
#print(label.shape)
if FullChannel: #去除缺通道的正常数据
    print("No Normal and other channel-missing data.")
    data=data[label!=0]
    label=label[label!=0]-1
    data=data[label<100]
    label=label[label<100]
else:
    print("No Normal data." if IgnoreNormal else "With Normal data.")
    if IgnoreNormal:
        data=data[label!=0]
        label=label[label!=0]-1
    label[label>=100]-=100
class_num = torch.unique(label).shape[0]
print(class_num)
print("Nunber of classes:{}.".format(class_num))

data_train,data_test,label_train,lable_test = train_test_split(data,label,train_size=0.7)
print(data_train.size())
# 自定义dataset和数据集预处理
preprocess = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=0, std=1)
])
class BearFaultDataset(Dataset):
    def __init__(self, inputs, targets, transform, reshape):
        if not FullChannel:
            #print(inputs.size())
            inputs = torch.split(inputs,1,1)[1]
            #print(inputs.size())
        inputs_f=torch.abs(torch.fft.fft(inputs))
        if reshape:
            '''这里现在有用了'''
            #self.inputs = torch.cat((torch.unsqueeze(inputs,1),torch.unsqueeze(inputs_f,1)),1)
            self.inputs = torch.unsqueeze(inputs_f,1)
            print(self.inputs.size())
            self.inputs = torch.squeeze(self.inputs)
            self.inputs = self.inputs[:, :2048]
            print("self.inputs",self.inputs.size())
            #self.inputs = torch.resize(self.inputs[:, :2025],(-1, 45, 45))
            self.inputs = torch.reshape(self.inputs,(inputs.shape[0], 32, 64))
        else:
            self.inputs = torch.cat((torch.unsqueeze(inputs,1),torch.unsqueeze(inputs_f,1)),1)
            #self.inputs = torch.unsqueeze(inputs,1)
            #self.inputs = torch.cat((torch.unsqueeze(inputs,1),torch.unsqueeze(inputs_f,1)),1)
        print("self.inputs size : ",self.inputs.size())
        self.targets = targets
        #print(self.targets.size())
        values = torch.unique(self.targets)
        print(values)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        #print('item = ',item)
        input = self.inputs[item]
        target = self.targets[item]
        # input = self.transform(input)
        return input, target
# 实例化dataset
isreshape = False
training_data = BearFaultDataset(data_train, label_train, transform=preprocess, reshape=isreshape)
#print(training_data)
test_data = BearFaultDataset(data_test, lable_test, transform=preprocess, reshape=isreshape)
print(training_data.inputs.shape, test_data.inputs.shape)
# 定义dataloader
batch_size = 32
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True,drop_last = True)
#print(train_dataloader)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True,drop_last = True)
ViT_Channels=3 if FullChannel else 1
print("Nunber of ViT channels:{}.".format(ViT_Channels))

v = Only_LSTM_Function.LSTMRNN( #有两个模型都可以选用，见LSTM_Funtion
    input_size = 2048,#input_size需要与LSTM输入X张量.size()的最后一维相同 X.size() = [64,2,2048] 
    hidden_size = 128, #LSTMRNN 中隐藏层的数量，可调，貌似越大收敛越快()
    num_layers = 2, #LSTM层层数,可调,貌似越小收敛越快()
    num_classes = 10, #分类个数,及目标target标签种类,不能改
    dropout = 0.1
    
).to(device)
weight_decay = 2e-4
epochs = 200 #定义训练轮数
# Initialize the loss function
loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 2e-3 #定义学习率
optimizer = torch.optim.NAdam(v.parameters(), lr=learning_rate,weight_decay=weight_decay,momentum_decay=9e-4) #定义优化器
ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1) #绑定衰减学习率到优化器
# 定义训练循环

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        #print(X.size())  #此时X.size() = [64,2,1,2048],1这个维度没啥用，用squeeze函数去掉
        X = torch.squeeze(X,2)
        #print(X.size())  #此时X.size() = [64,2,2048]，需要令input_size = 2048才可
        #X = X.permute(0, 1, 2)  #这个换位函数没啥用
        
        pred = model(X)  #X.size()=[batch_size,seq_length,input_size]
        #print(pred.size())
        #hidden_h = hidden_h.data
        #hidden_c = hidden_c.data
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
acc = []    
# 定义测试循环
def test_loop(dataloader, model, loss_fn,epoch,acc):
    hidden_h = None
    hidden_c = None
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    #更改模型为测试或者验证模式
    #model.eval()#把training属性设置为false,使模型处于测试或验证状态
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            X = torch.squeeze(X,2)
            #print(X.size())
            #X = X.permute(0, 1, 2)
            pred = model(X)
            '''
            if(epoch == 9):          
                np.save('tsne_test.npy',pred)
                np.save('tsne_acc.npy',y)
            '''
            #print(y.size())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")
    acc.append(100*correct)
    return test_loss,acc
   
last_loss=100
now_loss=100
for t in range(epochs): # 开始训练
    new_lr=ExpLR.get_last_lr()[0]
    #new_lr = learning_rate
    print(f"Epoch {t+1}\n-------------------------------")
    print(f'lr: {new_lr:>7e}')
    train_loop(train_dataloader, v, loss_fn, optimizer)
    last_loss=now_loss
    now_loss,acc=test_loop(test_dataloader, v, loss_fn,t,acc)
    
    # 学习率动态衰减
    if last_loss/now_loss <0.8:
        ExpLR.step()
        ExpLR.step()
    if last_loss/now_loss <0.9:
        ExpLR.step()
        ExpLR.step()
    if last_loss/now_loss <1.0:
        ExpLR.step()
        ExpLR.step()
    if last_loss/now_loss <1.4:
        ExpLR.step()
    else:ExpLR.step()
    
best = round(max(acc),3)
for i in range(len(acc)):
    print(' epoch ',i+1,'   ',' accuracy ',round(acc[i],3))
print("the best accuracy is ",best," in ",acc.index(max(acc))+1," epoch ")
print("Done!")

# 显示参数数量
nb_param = 0
for param in v.parameters():
    nb_param += np.prod(list(param.data.size()))
#for param in v.parameters():
#    print(type(param.data), param.size())
print('Number of parameters:', nb_param)
pass