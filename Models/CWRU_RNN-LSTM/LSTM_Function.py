import torch
from torch import nn
from vit_pytorch import Transformer
from einops import repeat
from einops.layers.torch import Rearrange
# 设定训练用的设备
device = "cuda" if torch.cuda.is_available() else "cpu"
# 打印看一下
#print("Using {} device".format(device))

class LSTM(nn.Module):#这个类里有我当时解决不了的bug，没啥用QAQ
    def __init__(self, input_size=2048, hidden_layer_size=128, output_size=10,batch_size = 32,batch_first = True):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.input_size = input_size
        # 创建LSTM层和linear层，LSTM层提取特征，linear层用作最后的预测
        ##LSTM算法接受三个输入：先前的隐藏状态，先前的单元状态和当前输入。
        self.lstm = nn.LSTM(input_size, hidden_layer_size,batch_first = True)
        self.linear = nn.Linear(hidden_layer_size*2, output_size)

        #初始化隐含状态及细胞状态C，hidden_cell变量包含先前的隐藏状态和单元状态
        self.hidden_cell = (torch.zeros(1, self.batch_size, self.hidden_layer_size),
                            torch.zeros(1, self.batch_size, self.hidden_layer_size))
                            

    def forward(self, input_seq):
        #print(input.shape)
        #a = input_seq.view(1, len(input_seq), -1)
        #print(a.size())
        #a = a.resize_(1,self.batch_size,self.input_size)
        #print(a.size())
        #print(input_seq.size())
        #input_seq = torch.squeeze(input_seq)
        #input_seq = input_seq.view()
        #input_seq = input_seq.permute(2, 1, 0)
        #print(input_seq.size())
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        #lstm的输出是当前时间步的隐藏状态ht和单元状态ct以及输出lstm_out
        #按照lstm的格式修改input_seq的形状，作为linear层的输入
        predictions = self.linear(lstm_out.reshape(len(input_seq), -1))
        return predictions#返回predictions的最后一个元素

class LSTMRNN(nn.Module):#这个是没有加池化层的原始LSTMRNN
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 初始化的隐藏元和记忆元,通常它们的维度是一样的
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #x.size(0)是batch_size
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class mylstm(nn.Module):#这个加了池化层，花里胡哨，效果也没多好QAQ
    def __init__(self,input_size=2048,hidden_size=128,num_layers=2, dropout=0.1,batch_size=64):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        super(mylstm, self).__init__()
        self.modle1 = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.modle2=nn.Sequential     (
            nn.Flatten(),
            nn.Linear(self.hidden_size*2,256),
            nn.ReLU(),
            #nn.Dropout(0.05),
            nn.Linear(256,10)
        )
    def forward(self, x):
        h_0 = torch.randn(self.num_layers,self.batch_size,self.hidden_size)
        c_0 = torch.randn(self.num_layers,self.batch_size,self.hidden_size)
        h_0=h_0.to(device)
        c_0=c_0.to(device)
        #x = x.to(torch.float32)
        x,(h_0,c_0)=self.modle1(x,(h_0,c_0))
        #x=x[:,-1,:]
        x=self.modle2(x)
        return x
