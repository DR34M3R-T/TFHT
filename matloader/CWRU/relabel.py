import numpy as np
from sklearn.neighbors import DistanceMetric

label_orig = np.load('dataset/CWRU/label.npy')
label_values = np.unique(label_orig)

#rpm
rpm = label_orig%10000
rpm_values = np.unique(rpm)
rpm3 = np.greater(rpm,np.full_like(rpm,1742))*20
rpm2 = np.greater(rpm,np.full_like(rpm3/1750*rpm,1760))*22
rpm1 = np.greater(rpm,np.full_like(rpm2/1772*rpm,1780))*25
rpm_fitted = rpm1+rpm2+rpm3+1730

#fan & drive & normal
fault_type = label_orig//10000000%10
fault_type_values = np.unique(fault_type)

#fault position
position = label_orig//1000000%10
position_values = np.unique(position)

#outer position
outer_position = label_orig//10000%100
outer_position_values = np.unique(outer_position)

#diameter
diameter = label_orig//1000000000%10*7
diameter_values = np.unique(diameter)

#channel
channel_tmp = label_orig//100000000%10
channel_values = np.unique(channel_tmp)
no_BA = np.greater(channel_tmp,0)
no_FE = np.greater(channel_tmp,2)

relabel = np.swapaxes(np.vstack((fault_type,position,diameter,outer_position,rpm_fitted,no_BA,no_FE)),0,1)
np.save('dataset/CWRU/relabel.npy',relabel)
pass