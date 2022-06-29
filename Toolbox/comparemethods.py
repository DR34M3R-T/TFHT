import torch
from torch import nn
torch.set_default_tensor_type(torch.FloatTensor)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from torch.optim.lr_scheduler import StepLR

# 读入excel数据以及保存部分，MFPT数据集
# df = pd.read_excel('./data/MFPTdataset.xlsx', header=None)
# orgdata = df.values
# print(type(orgdata))
# print(orgdata.shape)
# np.save('orgdata', orgdata)

# 重新载入MFPT数据集
# orgdata = np.load('./data/orgdata.npy')
# # orgdata = torch.from_numpy(orgdata)
# x = orgdata[:, :-1]
# y = orgdata[:, -1]
# # 输入数据标准化
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# 西储大学的数据集
# x_train = torch.from_numpy(np.load('./data/CasWestenData/x_train_pre.npy'))
# y_train = torch.from_numpy(np.load('./data/CasWestenData/y_train_pre.npy'))
# x_test = torch.from_numpy(np.load('./data/CasWestenData/x_valid_pre.npy'))
# y_test = torch.from_numpy(np.load('./data/CasWestenData/y_valid_pre.npy'))
# x_train = torch.from_numpy(np.load('./data/UCOON/xTrain.npy'))
# y_train = torch.from_numpy(np.load('./data/UCOON/yTrain.npy'))
# x_test = torch.from_numpy(np.load('./data/UCOON/xTest.npy'))
# y_test = torch.from_numpy(np.load('./data/UCOON/yTest.npy'))
x_train = torch.from_numpy(np.load('./dataset/XJTU/xTrain.npy'))
y_train = torch.from_numpy(np.load('./dataset/XJTU/yTrain.npy'))
x_test = torch.from_numpy(np.load('./dataset/XJTU/xTest.npy'))
y_test = torch.from_numpy(np.load('./dataset/XJTU/yTest.npy'))

# 自定义dataset和数据预处理
preprocess = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=0, std=1)
])

class BearFaultDataset(Dataset):
    def __init__(self, inputs, targets, transform, reshape):
        if reshape:
            self.inputs = inputs[:, :2025].reshape((-1, 45, 45))
        else:
            self.inputs = inputs
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
print(training_data.inputs.shape, test_data.inputs.shape)
# 定义dataloader
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.float().to(device), y.long().to(device)

        # Compute prediction error
        pred, _ = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, train=False):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.float().to(device), y.long().to(device)
            pred, _ = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    avg_test_loss = test_loss / size
    correct /= size
    if train:
        print(f"Train Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {avg_test_loss:>8f}, "
              f"Total loss: {test_loss:>8f} \n")
    else:
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {avg_test_loss:>8f}, "
              f"Total loss: {test_loss:>8f} \n")
    return np.array(avg_test_loss), np.array(correct)

# 设定训练用的设备
device = "cuda" if torch.cuda.is_available() else "cpu"
# 打印看一下
print("Using {} device".format(device))

# 定义用来做对比实验的模型
class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=128, stride=128)
        )
        self.lstm = nn.LSTM(
            input_size = 45,
            hidden_size = 64,
            num_layers = 5,
            batch_first = True,
            dropout = 0.1
        )
        self.to_latent = nn.Sequential(
            nn.Linear(64, out_features = 128),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.out = nn.Sequential(
            nn.Linear(128, out_features = 4),
            nn.ReLU()
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        # x = self.to_patch_embedding(x).transpose(1, 2)
        lstm_out, (h_n, h_c) = self.lstm(x)
        features = self.to_latent(lstm_out[:, -1, :])
        return self.out(features), features


class MyCNN1D(nn.Module):
    def __init__(self):
        super(MyCNN1D, self).__init__()
        self.flatten = nn.Flatten()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=128, stride=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.to_latent = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU()
        )

        self.out = nn.Linear(100, 4)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.flatten(x)
        features = self.to_latent(x)
        return self.out(features), features


epochs = 10
test_acc_endtrain_list = []
for times in range(1, 2, 1):
    print(f"Training Times {times}\n")
    # 定义模型ViT并实例化
    # model = MyLSTM().to(device)
    model = MyCNN1D().to(device)
    total = sum(param.nelement() for param in model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = AMSoftmax(in_feats = 1024, n_classes = 10)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # scheduler
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    train_loss_list, train_acc_list = [], []
    test_loss_list, test_acc_list = [], []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        # scheduler.step()
        train_loss, train_acc = test(train_dataloader, model, loss_fn, train=True)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss, test_acc = test(test_dataloader, model, loss_fn, train=False)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
    print("Done!")
    test_acc_endtrain_list.append(test_acc)
    # 保存训练好的网络
    torch.save(model.state_dict(), './result/pretrained-net_{}.pt'.format(times))
    np.save('./result/train_loss_{}'.format(times), train_loss_list)
    np.save('./result/train_acc_{}'.format(times), train_acc_list)
    np.save('./result/test_loss_{}'.format(times), test_loss_list)
    np.save('./result/test_acc_{}'.format(times), test_acc_list)
np.save('./result/test_acc_endtrain_list', test_acc_endtrain_list)
