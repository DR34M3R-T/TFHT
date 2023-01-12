#TSNE 示例代码
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

mpl.rcParams['font.family'] = ['Times New Roman']
mpl.rcParams["axes.titlepad"] = 16
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markersize'] = 4
mpl.rcParams['lines.markeredgecolor'] = 'black'
mpl.rcParams['lines.markeredgewidth'] = '0.5'
mpl.rcParams['xtick.bottom'] = False
mpl.rcParams['ytick.left'] = False


# 加载数据
def get_data():
	"""
	:return: 数据集、标签、样本数量、特征数量
	"""
	digits = datasets.load_digits(n_class=10)
	data = digits.data		# 图片特征
	label = digits.target		# 图片标签
	n_samples, n_features = data.shape		# 数据集的形状
	return data, label, n_samples, n_features

# 绘制置信椭圆
def confidence_ellipse(x, y, ax, color='auto', n_std=3.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = mpl.patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=color, alpha=0.2, **kwargs)
    ellipse_edge = mpl.patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      edgecolor=color, facecolor='#FFFFFF00', **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = mpl.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ellipse_edge.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse_edge),ax.add_patch(ellipse)
# 对样本进行预处理并画图
def plot_embedding(data, label, title, class_dict):
	"""
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
	class_num = np.unique(label).shape[0]
	#x_min, x_max = np.min(data, 0), np.max(data, 0)
	#data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
	fig = plt.figure()		# 创建图形实例
	ax = plt.subplot(111)		# 创建子图
	# 遍历所有样本
	for i in range(class_num):
		# 在图中为每个数据点画出标签
		data_class_i = data[label==i]
		ax = plt.plot(data_class_i[:,0],data_class_i[:,1],'o',label=class_dict[i],color=f'C{i}')
		confidence_ellipse(data_class_i[:,0],data_class_i[:,1],fig.axes[0],color=f'C{i}')
		# plt.text(data[i, 0], data[i, 1], str(label[i].item()), color=plt.cm.Set1(label[i].item() / 10),
		#			fontdict={'weight': 'bold', 'size': 7})
	plt.xticks()		# 指定坐标的刻度
	plt.yticks()
	plt.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=10)#, mode='expand')
	plt.title(title, fontsize=14)
	# 返回值
	return fig

# data, label , n_samples, n_features = get_data()		# 调用函数，获取数据集信息

data=np.load('./result/XJTU/CNN_tsne_npy/feature.npy')#[:1000]
label=np.load('./result/XJTU/CNN_tsne_npy/label.npy')#[:1000]

pd10_classlist=[
    'NC','B007','B014','B021',
    'IR007','IR014','IR021',
    'OR007','OR014','OR021',
]

xj_classlist=[
    'OR','IBCO','IR','Cage'
]

p4_classlist=[
    'NC','B','IR','OR']

print('Starting compute t-SNE Embedding...')
ts = TSNE(n_components=2, learning_rate=200, init='pca')
# t-SNE降维
result = ts.fit_transform(data)
# 调用函数，绘制图像
fig = plot_embedding(result, label, 't-SNE Embedding of CNN1D features', xj_classlist)
# 显示图像
plt.show()
pass
