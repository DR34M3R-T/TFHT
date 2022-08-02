import datetime
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import numpy as np
import MyViT_CWRU
import logging

def auto_train(argvs):
    (label_name, FullChannel, 
    IgnoreNormal, batch_size, patch_size, 
    dim, depth, head, dim_head, mlp_dim) = argvs
    
    # 开始时间
    start_time=datetime.datetime.now()
    # 设定为DoubleTensor
    torch.set_default_tensor_type(torch.DoubleTensor)
    # 设定训练用的设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # FullChannel=False
    # IgnoreNormal=False

    # 导入raw数据集
    # label_name='label_end&position_7.npy'
    data = torch.from_numpy(np.load('./dataset/CWRU/data.npy')) #7253 3 2048
    label = torch.from_numpy(np.load('./dataset/CWRU/'+label_name)) #7253
    label = label.type(torch.LongTensor)

    # 去除负标签
    data=data[label!=-1]
    label=label[label!=-1]
    if FullChannel: #去除缺通道的正常数据
        data=data[label!=0]
        label=label[label!=0]-1
        data=data[label<100]
        label=label[label<100]
    else:
        if IgnoreNormal:
            data=data[label!=0]
            label=label[label!=0]-1
        label[label>=100]-=100
    class_num = torch.unique(label).shape[0]
    ViT_Channels=3 if FullChannel else 1

    #重定向log输出
    logname = label_name[:-4] + '/' + \
        'C' + str(ViT_Channels) + \
        '-cn' + str(class_num) + \
        '-bs' + str(batch_size) + \
        '-ps' + str(patch_size) + \
        '-d' + str(dim) + \
        '-dp' + str(depth) + \
        '-h' + str(head) + \
        '-dk' + str(dim_head) + \
        '-md' + str(mlp_dim) +'.log'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(filename='./result/CWRU/logs/{}'.format(logname), mode='a', encoding="utf-8", delay=False)
    fmt = logging.Formatter("[%(asctime)s] %(message)s")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt=fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("## start time: {}".format(start_time))
    logger.info("Using {} device".format(device))
    logger.info("In train:{}.".format(label_name))
    logger.info("Full Channel" if FullChannel else "One Channel")
    logger.info("No Normal and other channel-missing data." if FullChannel else (
        "No Normal data." if IgnoreNormal else "With Normal data."))
    logger.info("Nunber of classes:{}.".format(class_num))
    logger.info("Nunber of ViT channels:{}.".format(ViT_Channels))

    rand_arr = np.random.randint(low=0, high=10, size=data.shape[0])
    rand_arr = torch.from_numpy(np.bool_(np.clip(rand_arr,2,3)-2))
    data_train = data[rand_arr==1]
    label_train = label[rand_arr==1]
    data_test = data[rand_arr==0]
    lable_test = label[rand_arr==0]

    # 自定义dataset和数据集预处理
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=0, std=1)
    ])
    class BearFaultDataset(Dataset):
        def __init__(self, inputs, targets, transform, reshape):
            if not FullChannel:
                inputs = torch.split(inputs,1,1)[1]
            inputs_f=torch.abs(torch.fft.fft(inputs))
            #inputs_f/=len(inputs_f[0])/2
            #inputs_f[0]/=2
            #logger.info(inputs.shape)
            #logger.info(inputs_f.shape)
            if reshape:
                '''这里还没写完qaq不过好像也没啥用'''
                self.inputs = torch.cat(torch.unsqueeze(inputs,1),torch.unsqueeze(inputs_f,1))
                self.inputs = self.inputs[:, :2025].reshape((-1, 45, 45))
            else:
                self.inputs = torch.cat((torch.unsqueeze(inputs,1),torch.unsqueeze(inputs_f,1)),1)
            self.targets = targets
            values = torch.unique(self.targets)
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
    training_data = BearFaultDataset(data_train, label_train, transform=preprocess, reshape=isreshape)
    test_data = BearFaultDataset(data_test, lable_test, transform=preprocess, reshape=isreshape)
    # logger.info(training_data.inputs.shape, test_data.inputs.shape)
    # 定义dataloader
    # batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    v = MyViT_CWRU.ViT( #定义ViT模型
        image_size = 2048,
        patch_size = patch_size,
        channels=ViT_Channels,
        num_classes = class_num,
        dim = dim,
        depth = depth,
        heads = head,
        dim_head = dim_head,
        mlp_dim = mlp_dim,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)#这里的训练强度已经减小了
    epochs = 10 #定义训练轮数

    # 加载模型
    # v=torch.load('./result/ViT-pretrained-net.pt')

    # Initialize the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 0.002 #定义学习率
    optimizer = torch.optim.NAdam(v.parameters(), lr=learning_rate,weight_decay=4e-5,momentum_decay=9e-4) #定义优化器
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) #绑定衰减学习率到优化器

    # 定义训练循环
    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    # 定义测试循环
    def test_loop(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        logger.info(f"Test Error: Accuracy: {(100*correct):>0.3f}%, Avg loss: {test_loss:>8f}")
        return test_loss

    last_loss=100
    now_loss=100
    for t in range(epochs): # 开始训练
        new_lr=ExpLR.get_last_lr()[0]
        logger.info(f"Epoch {t+1}---------------")
        logger.info(f'lr: {new_lr:>7e}')
        train_loop(train_dataloader, v, loss_fn, optimizer)
        last_loss=now_loss
        now_loss=test_loop(test_dataloader, v, loss_fn)
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
    logger.info("Done!")

    torch.save(v.state_dict(), './result/ViT-state.pt') # 保存训练的模型


    # 显示参数数量
    nb_param = 0
    for param in v.parameters():
        nb_param += np.prod(list(param.data.size()))
    #for param in v.parameters():
    #    logger.info(type(param.data), param.size())
    logger.info('Number of parameters:', nb_param)

# label_name, FullChannel, 
# IgnoreNormal, batch_size, patch_size, 
# dim, depth, head, dim_head, mlp_dim
iter_list = [
    #['p&d10.npy',False,False,64,8,128,6,6,64,256],
    #['p&d10.npy',False,False,64,16,128,6,6,64,256],
    #['p&d10.npy',False,False,64,32,128,6,6,64,256],
    #['p&d10.npy',False,False,64,64,128,6,6,64,256],
    ['p&d10.npy',False,False,64,128,128,6,6,64,256],
    ['p&d10.npy',False,False,64,256,128,6,6,64,256],
    ['p&d10.npy',False,False,64,512,128,6,6,64,256],
    ['p&d10.npy',False,False,64,1024,128,6,6,64,256],
    ['p&d10.npy',False,False,64,2048,128,6,6,64,256],
    ['p&d10.npy',False,False,64,8,16,6,6,64,32],
    ['p&d10.npy',False,False,64,8,32,6,6,64,64],
    ['p&d10.npy',False,False,64,8,64,6,6,64,128],
    ['p&d10.npy',False,False,64,8,128,6,6,8,256],
    ['p&d10.npy',False,False,64,8,128,6,6,16,256],
    ['p&d10.npy',False,False,64,8,128,6,6,32,256],
    ['p&d10.npy',False,False,64,8,128,6,6,128,256],
    ['p&d10.npy',False,False,64,8,128,6,1,64,256],
    ['p&d10.npy',False,False,64,8,128,6,2,64,256],
    ['p&d10.npy',False,False,64,8,128,6,4,64,256],
    ['p&d10.npy',False,False,64,8,128,1,6,64,256],
    ['p&d10.npy',False,False,64,8,128,2,6,64,256],
    ['p&d10.npy',False,False,64,8,128,4,6,64,256]
]
l = len(iter_list)
for i in range(l):
    argvs=iter_list[i]
    auto_train(argvs)
    print(f'{i+1} of {l} finished.')