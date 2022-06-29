#可视化示例代码

import matplotlib.pyplot as plt
import numpy as np

#导入数据
x_train = np.load('./dataset/XJTU/xTrain.npy')
y_train = np.load('./dataset/XJTU/yTrain.npy')
x_test = np.load('./dataset/XJTU/xTest.npy')
y_test = np.load('./dataset/XJTU/yTest.npy')

#fft
x_train_FFT = np.abs(np.fft.fft(x_train))
x_train_FFT = x_train_FFT/1024 #fft振幅标准化
x_train_FFT[0]/=2 #fft振幅标准化

ypoints=np.arange(2048)

for i in range(8):
    xpoints=x_train[0]
    title=y_train[0]
    plt.title(f'x_train[{i}] | type{title}')
    plt.xlim(200*i,200*i+200)
    plt.plot(ypoints, xpoints)
    plt.savefig("pics/samples/time_input_cut{}.png".format(i))
    plt.show()

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

'''
for i in range(2800):
    xpoints=x_train[i]
    title=y_train[i]
    plt.title(f'x_train[{i}] | type{title}')
    #plt.xlim(0,2000)
    #plt.ylim(-4,4)
    plt.plot(ypoints, xpoints)
    plt.show()
'''