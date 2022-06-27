import scipy.io as sio
import numpy as np
import os

def matloader(file_name,path):
    #load
    data_to_add = np.load('dataset/CWRU/data.npy')
    lable_to_add = np.load('dataset/CWRU/lable.npy')

    #file_name = "OR007@3_0.mat"
    #path = 'dataset/12k_fan_end/'
    #path = path +'/'
    data = sio.loadmat(path + file_name)

    channels = list(data.keys())
    channels.remove('__header__')
    channels.remove('__version__')
    channels.remove('__globals__')
    try:
        channels.remove('i')
    except:pass

    lable={
        'fault':''
    }
    lable_code=0

    if (path[-7:-1] == 'normal'):
        lable['fault'] = '0'
    else: 
        if (path[-10:-1] == 'drive_end'):
            lable['fault'] = '1'
            lable_code += 1*10000000
        elif (path[-8:-1] == 'fan_end'):
            lable['fault'] = '2'
            lable_code += 2*10000000

        if (file_name[0:1] == 'B'):
            lable['position'] = 'ball'
            lable['d'] = file_name[2:4]
            lable_code += 1*1000000
        elif (file_name[0:2] == 'IR'):
            lable['position'] = 'inner'
            lable['d'] = file_name[3:5]
            lable_code += 2*1000000
        elif (file_name[0:2] == 'OR'):
            lable['position'] = 'outer'
            lable_code += 3*1000000
            lable['d'] = file_name[3:5]
            if (file_name[-7:-6] == '2'):
                lable['outer_position'] = '12'
                lable_code += 12*10000
            elif(file_name[-7:-6] == '3'):
                lable['outer_position'] = '03'
                lable_code += 3*10000
            elif(file_name[-7:-6] == '6'):
                lable['outer_position'] = '06'
                lable_code += 6*10000

    x_name = channels[0][0:4]
    try:
        rpm=int( np.squeeze(data[x_name + 'RPM']) )
    except:rpm=1797
    lable['rpm'] = rpm
    lable_code += rpm

    try:
        base=np.squeeze(data[x_name + '_BA_time'])
    except:
        base=np.zeros((300000,))
    try:
        fan=np.squeeze(data[x_name + '_FE_time'])
    except:
        fan=np.zeros((300000,))
    try:
        drive=np.squeeze(data[x_name + '_DE_time'])
    except:
        drive=np.zeros((300000,))

    #cut!
    data_cut = np.empty([int(drive.size/2048),3,2048])
    lable_cut =[]
    i = 0
    while((i+1)<drive.size/2048):
        tmp = np.empty([3,2048])
        tmp[0] = fan[i*2048:(i+1)*2048]
        tmp[1] = drive[i*2048:(i+1)*2048]
        tmp[2] = base[i*2048:(i+1)*2048]
        data_cut[i] = tmp
        lable_cut.append(lable_code)
        i+=1
        pass
    lable_cut = np.array(lable_cut)
    lable_cut_combine = np.concatenate((lable_cut,lable_to_add))
    data_cut_combine = np.concatenate((data_cut,data_to_add))

    np.save('dataset/CWRU/lable.npy',lable_cut_combine)
    np.save('dataset/CWRU/data.npy',data_cut_combine)
    print('finish')
    pass

filepath = 'dataset/12k_drive_end/'
filelist = []
# 遍历filepath下所有文件，包括子目录
files = os.listdir(filepath)
for fi in files:
    fi_d = os.path.join(filepath, fi)
    if not (os.path.isdir(fi_d)):
        filelist.append(fi)
for item in files:
    if ((item!='rename.bat') & (item!='renamelist.xlsx') & (item!='OR007@3_0.mat')): matloader(item,filepath)