#coding=utf-8
__author__ = "Hai Wang"

import scipy.io as sio 
import numpy as np

def Feature_merge_tree():
	A = sio.loadmat('data/Data_tree_v.mat')
	Train_data_t = A['Train_data']
	Train_label = A['Train_Y']
	Test_data_t = A['Test_data']
	Unlabel_data_t = A['Unlabel_data']

	B = sio.loadmat('data/Data_group.mat')
	Train_data_g = B['Train_data']
	Test_data_g = B['Test_data']
	Unlabel_data_g = B['Unlabel_data']



	Train_data = np.hstack((Train_data_t,Train_data_g));
	Test_data = np.hstack((Test_data_t,Test_data_g));
	if len(Unlabel_data_t)>0:
		Unlabel_data = np.hstack((Unlabel_data_t,Unlabel_data_g));

	print('Number of features is ' + str(Train_data.shape[1]))
	sio.savemat('data/Data_tree.mat', {'Train_data': Train_data,'Train_Y': Train_label,'Test_data':Test_data,'Unlabel_data':Unlabel_data})

