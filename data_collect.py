import sys
import imp
x=10
import comparemethods
for i in range(x):
    #start
    output = sys.stdout
    outputfile = open('result/pics/CNN1D/test{}.txt'.format(i+1),'w+')
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
    outputfile = open('result/pics/ViT/test{}.txt'.format(i+1),'w+')
    sys.stdout = outputfile
    imp.reload(ViT_torch_train_rotor)
    #end
    outputfile.close()
    sys.stdout = output
    print("ViT {} of {}".format(i+1,x))

