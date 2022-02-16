import MyViT
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
import random
# from ViT_torch_train_rotor import BearFaultDataset
# 定义并加载预训练的模型
model = MyViT.ViT(
    image_size = 2048,
    patch_size = 64,
    num_classes = 4,
    dim = 64,
    depth = 2,
    heads = 4,
    mlp_dim = 128,
    dropout = 0.1,
    emb_dropout = 0.1
) 
model.load_state_dict(torch.load('./result/ViT-state.pt'))
model.eval()

# 加载数据
data=torch.from_numpy(np.load('./dataset/XJTU/xTrain.npy'))
lable=torch.from_numpy(np.load('./dataset/XJTU/yTrain.npy'))
# 随机取出一条
# num = random.choice(range(2800))

for num in [1336]:#range(len(data)):
    data0=data[num].unsqueeze(0)
    data0 = torch.cat((torch.unsqueeze(data0,1),torch.unsqueeze(torch.abs(torch.fft.fft(data0)),1)),1)
    lable0=lable[num]
    with torch.no_grad():
        out = model(data0)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    categories = [
        'type0',
        'type1',
        'type2',
        'type3'
    ]
    # Print top categories per image
    top_prob, top_catid = torch.topk(probabilities, 4)
    print('Data[{}] expect lable: type{}'.format(num,lable0.item()))
    print('Test result:')
    for i in range(top_prob.size(0)):
        print(categories[top_catid[i]], f'{top_prob[i].item():>.5f}')