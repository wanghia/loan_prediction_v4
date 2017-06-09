#coding=utf-8
__author__ = "Hai Wang"

import scipy.io as sio 
import numpy as np

def Feature_merge_linear():
	A = sio.loadmat('data/Data_Source.mat')
	Train_data_O = A['Train_data']
	Train_label = A['Train_Y']
	Test_data_O = A['Test_data']
	Unlabel_data_O = A['Unlabel_data']

	B = sio.loadmat('data/Data_Location.mat')
	Train_data_L = B['Train_data']
	Test_data_L = B['Test_data']
	Unlabel_data_L = B['Unlabel_data']



	top_len = Train_data_L.shape[1]
	cur_top = 400
	top = [i for i in range(min([cur_top,top_len]))]

	Train_data = np.hstack((Train_data_O,Train_data_L[:,top]));
	Test_data = np.hstack((Test_data_O,Test_data_L[:,top]));
	if len(Unlabel_data_O)>0:
		Unlabel_data = np.hstack((Unlabel_data_O,Unlabel_data_L[:,top]));

	print('Number of features is ' + str(Train_data.shape[1]))
	sio.savemat('data/Data_linear.mat', {'Train_data': Train_data,'Train_Y': Train_label,'Test_data':Test_data,'Unlabel_data':Unlabel_data})

