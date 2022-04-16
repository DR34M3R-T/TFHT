import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import numpy as np
import MyViT

# 设定训练用的设备
device = "cuda" if torch.cuda.is_available() else "cpu"
# 打印看一下
print("Using {} device".format(device))

# 导入raw数据集
x_train = torch.from_numpy(np.load('./dataset/XJTU/xTrain.npy'))
y_train = torch.from_numpy(np.load('./dataset/XJTU/yTrain.npy'))
x_test = torch.from_numpy(np.load('./dataset/XJTU/xTest.npy'))
y_test = torch.from_numpy(np.load('./dataset/XJTU/yTest.npy'))

# 自定义dataset和数据集预处理
preprocess = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=0, std=1)
])
class BearFaultDataset(Dataset):
    def __init__(self, inputs, targets, transform, reshape):
        inputs_f=torch.abs(torch.fft.fft(inputs))
        #inputs_f/=len(inputs_f[0])/2
        #inputs_f[0]/=2
        #print(inputs.shape)
        #print(inputs_f.shape)
        if reshape:
            '''这里还没写完qaq不过好像也没啥用'''
            self.inputs = torch.cat(torch.unsqueeze(inputs,1),torch.unsqueeze(inputs_f,1))
            self.inputs = self.inputs[:, :2025].reshape((-1, 45, 45))
        else:
            self.inputs = torch.cat((torch.unsqueeze(inputs,1),torch.unsqueeze(inputs_f,1)),1)
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        input = self.inputs[item]
        target = self.targets[item]
        # input = self.transform(input)
        return input, target


# 实例化dataset
isreshape = False
training_data = BearFaultDataset(x_train, y_train, transform=preprocess, reshape=isreshape)
test_data = BearFaultDataset(x_test, y_test, transform=preprocess, reshape=isreshape)
# print(training_data.inputs.shape, test_data.inputs.shape)
# 定义dataloader
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

v = MyViT.ViT( #定义ViT模型
    image_size = 2048,
    patch_size = 64,
    num_classes = 4,
    dim = 64,
    depth = 2,
    heads = 4,
    mlp_dim = 128,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)#这里的训练强度已经减小了
epochs = 2 #定义训练轮数

# 加载模型
# v=torch.load('./result/ViT-pretrained-net.pt')

# Initialize the loss function
loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 0.0008 #定义学习率
optimizer = torch.optim.Adam(v.parameters(), lr=learning_rate) #定义优化器
ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8) #绑定衰减学习率到优化器

# 定义训练循环
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
# 定义测试循环
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

last_loss=100
now_loss=100
for t in range(epochs): # 开始训练
    new_lr=ExpLR.get_last_lr()[0]
    print(f"Epoch {t+1}\n-------------------------------")
    print(f'lr: {new_lr:>7e}')
    train_loop(train_dataloader, v, loss_fn, optimizer)
    last_loss=now_loss
    now_loss=test_loop(test_dataloader, v, loss_fn)
    # 学习率动态衰减
    if last_loss/now_loss <0.7:
        ExpLR.step()
    if last_loss/now_loss <0.85:
        ExpLR.step()
    if last_loss/now_loss <1:
        ExpLR.step()
print("Done!")

torch.save(v.state_dict(), './result/ViT-state.pt') # 保存训练的模型


# 显示参数数量
nb_param = 0
for param in v.parameters():
    nb_param += np.prod(list(param.data.size()))
#for param in v.parameters():
#    print(type(param.data), param.size())
print('Number of parameters:', nb_param)