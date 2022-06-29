#数据收集脚本
import sys
import imp
x=5
'''
import comparemethods
for i in range(x):
    #start
    output = sys.stdout
    outputfile = open('pics/CNN1D/test{}.txt'.format(i+1),'w+')
    sys.stdout = outputfile
    imp.reload(comparemethods)
    #end
    outputfile.close()
    sys.stdout = output
    print("CNN1D {} of {}".format(i+1,x))
    
import ViT_torch_train_rotor
for i in range(x):
    #start
    output = sys.stdout
    outputfile = open('pics/ViT/test{}.txt'.format(i+1),'w+')
    sys.stdout = outputfile
    imp.reload(ViT_torch_train_rotor)
    #end
    outputfile.close()
    sys.stdout = output
    print("ViT {} of {}".format(i+1,x))
'''

'''
from old import ViT_torch_train_rotor_without_FFT
for i in range(x):
    #start
    output = sys.stdout
    outputfile = open('pics/ViT_without_FFT/test{}.txt'.format(i+1),'w+')
    sys.stdout = outputfile
    imp.reload(ViT_torch_train_rotor_without_FFT)
    #end
    outputfile.close()
    sys.stdout = output
    print("ViT {} of {}".format(i+1,x))
'''

from old import ViT_torch_train_rotor_only_Freq
for i in range(x):
    #start
    output = sys.stdout
    outputfile = open('pics/ViT_only_Freq/test{}.txt'.format(i+1),'w+')
    sys.stdout = outputfile
    imp.reload(ViT_torch_train_rotor_only_Freq)
    #end
    outputfile.close()
    sys.stdout = output
    print("ViT {} of {}".format(i+1,x))
