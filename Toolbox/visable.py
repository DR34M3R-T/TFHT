#可视化示例代码

from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

#导入数据
x_train = np.load('./dataset/CWRU/data.npy')
y_train = np.load('./dataset/CWRU/label.npy')
x_test = np.load('./dataset/XJTU/xTest.npy')
y_test = np.load('./dataset/XJTU/yTest.npy')

label_uniq = np.unique(y_train)

#fft
x_train_FFT = np.abs(np.fft.fft(x_train))
x_train_FFT = x_train_FFT/1024 #fft振幅标准化
x_train_FFT[0]/=2 #fft振幅标准化

ypoints=[
    np.arange(2048),
    np.arange(2048),
    np.arange(2048)
    ]
ypoints = np.swapaxes(ypoints,0,1)
'''
for i in range(8):
    xpoints=x_train[0]
    title=y_train[0]
    plt.title(f'x_train[{i}] | type{title}')
    plt.xlim(200*i,200*i+200)
    plt.plot(ypoints, xpoints)
    plt.savefig("pics/samples/time_input_cut{}.png".format(i))
    plt.show()
'''
'''
for i in range(8):
    xpoints=x_train_FFT[0]
    title=y_train[0]
    plt.title(f'x_train_FFT[{i}] | type{title}')
    plt.xlim(200*i,200*i+200)
    plt.plot(ypoints, xpoints)
    plt.savefig("pics/samples/freq_input_cut{}.png".format(i))
    plt.show()
'''

for i in range(7253):
    xpoints=np.swapaxes(x_train[i],0,1)
    if(xpoints[0][0]==0.0):
        title=y_train[i]
        plt.title(f'x_train[{i}] | type{title}')
        #plt.xlim(0,2000)
        #plt.ylim(-4,4)
        plt.plot(ypoints, xpoints)
        plt.show()