import matplotlib as mpl
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['font.family'] = ['Times New Roman']
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markersize'] = 4
mpl.rcParams['xtick.bottom'] = False
mpl.rcParams['ytick.left'] = False

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm_num = cm
        cm = cm.astype('float') / cm.sum(axis=1)#[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                txt = '100' if cm[i,j]==1.0 else '{:.2f}'.format(cm[i,j]*100)
            else :
                txt = str(cm[i,j])
            plt.text(j, i, txt,
                 horizontalalignment="center",
                 verticalalignment="center", weight='bold',
                 color="white" if cm[i, j] > thresh else "black")
    # plt.tight_layout()
    plt.ylabel('True bearing fault mode')
    plt.xlabel('Predicted bearing fault mode')
    plt.show()

def gen_confusion_matrix(real,pred,class_list):
    class_num = np.unique(real).shape[0]
    cm = np.zeros((class_num,class_num),dtype=np.int32)
    for i in range(real.shape[0]):
        cm[int(real[i]),int(pred[i])]+=1
    # cm[0,9]=114514
    plot_confusion_matrix(cm,class_list,normalize=True)
    pass

real = np.load('./result/XJTU/conf_mat/real.npy')
pred = np.load('./result/XJTU/conf_mat/pred.npy')

pd10_classlist=[
    'NC','B007','B014','B021',
    'IR007','IR014','IR021',
    'OR007','OR014','OR021',
]
xj_classlist=[
    'OR','IBCO','IR','Cage'
]

gen_confusion_matrix(real,pred,xj_classlist)