import scipy.io as sio
import numpy as np

#load
#data_to_add = np.load('dataset/CWRU/data.npy')
#label_to_add = np.load('dataset/CWRU/label.npy')

file_name = "OR007@3_0.mat"
path = 'dataset/12k_fan_end/'
#path = path +'/'
data = sio.loadmat(path + file_name)

channels = list(data.keys())
channels.remove('__header__')
channels.remove('__version__')
channels.remove('__globals__')
try:
    channels.remove('i')
except:pass

label={
    'fault':''
}
label_code=0

if (path[-7:-1] == 'normal'):
    label['fault'] = '0'
else: 
    if (path[-10:-1] == 'drive_end'):
        label['fault'] = '1'
        label_code += 1*10000000
    elif (path[-8:-1] == 'fan_end'):
        label['fault'] = '2'
        label_code += 2*10000000

    if (file_name[0:1] == 'B'):
        label['position'] = 'ball'
        label['d'] = file_name[2:4]
        label_code += 1*1000000
        if (file_name[1:4] == '007'): label_code += 1*1000000000
        elif(file_name[1:4] == '014'):label_code += 2*1000000000
        elif(file_name[1:4] == '021'):label_code += 3*1000000000
        elif(file_name[1:4] == '028'):label_code += 4*1000000000
    elif (file_name[0:2] == 'IR'):
        label['position'] = 'inner'
        label['d'] = file_name[3:5]
        label_code += 2*1000000
        if (file_name[2:5] == '007'): label_code += 1*1000000000
        elif(file_name[2:5] == '014'):label_code += 2*1000000000
        elif(file_name[2:5] == '021'):label_code += 3*1000000000
        elif(file_name[2:5] == '028'):label_code += 4*1000000000
    elif (file_name[0:2] == 'OR'):
        label['position'] = 'outer'
        label_code += 3*1000000
        label['d'] = file_name[3:5]
        if (file_name[-7:-6] == '2'):
            label['outer_position'] = '12'
            label_code += 12*10000
        elif(file_name[-7:-6] == '3'):
            label['outer_position'] = '03'
            label_code += 3*10000
        elif(file_name[-7:-6] == '6'):
            label['outer_position'] = '06'
            label_code += 6*10000
        if (file_name[2:5] == '007'): label_code += 1*1000000000
        elif(file_name[2:5] == '014'):label_code += 2*1000000000
        elif(file_name[2:5] == '021'):label_code += 3*1000000000
        elif(file_name[2:5] == '028'):label_code += 4*1000000000
    

x_name = channels[0][0:4]
try:
    rpm=int( np.squeeze(data[x_name + 'RPM']) )
except:
    if((file_name == 'normal_0.mat') |
        (file_name == 'B028_0.mat') |
        (file_name == 'IR028_0.mat')
        ):
        rpm = 1797
    elif((file_name == 'normal_1.mat') |
        (file_name == 'B028_1.mat') |
        (file_name == 'IR028_1.mat')):
        rpm = 1772
    elif((file_name == 'normal_2.mat') |
        (file_name == 'B028_2.mat') |
        (file_name == 'IR028_2.mat')):
        rpm = 1750
    elif((file_name == 'normal_3.mat') |
        (file_name == 'B028_3.mat') |
        (file_name == 'IR028_3.mat')):
        rpm = 1730
    else:rpm = 1700
label['rpm'] = rpm
label_code += rpm

try:
    base=np.squeeze(data[x_name + '_BA_time'])
except:
    base=np.zeros((500000,))
    label_code += 1*100000000
try:
    fan=np.squeeze(data[x_name + '_FE_time'])
except:
    fan=np.zeros((500000,))
    label_code += 2*100000000
try:
    drive=np.squeeze(data[x_name + '_DE_time'])
except:
    drive=np.zeros((500000,))
    label_code += 4*100000000

if (file_name == 'normal_2.mat'):
    fan2=np.squeeze(data['X099_FE_time'])
    drive2=np.squeeze(data['X099_DE_time'])
    fan = np.split(fan,[483328])[0]
    drive = np.split(drive,[483328])[0]
    fan = np.concatenate((fan,fan2))
    drive = np.concatenate((drive,drive2))
    base=np.zeros((1000000,))
    pass

#cut!
data_cut = np.empty([int(drive.size/2048),3,2048])
label_cut =[]
i = 0
while((i+1)<drive.size/2048):
    tmp = np.empty([3,2048])
    tmp[0] = fan[i*2048:(i+1)*2048]
    tmp[1] = drive[i*2048:(i+1)*2048]
    tmp[2] = base[i*2048:(i+1)*2048]
    data_cut[i] = tmp
    label_cut.append(label_code)
    i+=1
    pass
label_cut = np.array(label_cut)
#label_cut_combine = np.concatenate((label_cut,label_to_add))
#data_cut_combine = np.concatenate((data_cut,data_to_add))

np.save('dataset/CWRU/label.npy',label_cut)
np.save('dataset/CWRU/data.npy',data_cut)
print('finish')
pass