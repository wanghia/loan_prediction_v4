#coding=utf-8
__author__ = "Hai Wang"

import scipy.io as sio 
import numpy as np

def Feature_merge_linear_NFS():
	A = sio.loadmat('data/Data_NFS_One_hot.mat')
	Train_data_O = A['Train_data']
	Train_data_Y = A['Train_Y']
	Test_data_O = A['Test_data']
	Unlabel_data_O = A['Unlabel_data']

	D = sio.loadmat('data/Data_NFS_scale.mat')
	Train_data_Sc = D['Train_data']
	Test_data_Sc = D['Test_data']
	Unlabel_data_Sc = D['Unlabel_data']
	Train_data_S = np.hstack(([Train_data_O,Train_data_Sc]))
	Test_data_S = np.hstack(([Test_data_O,Test_data_Sc]))
	if (len(Unlabel_data_Sc)>0):
		Unlabel_data_S = np.hstack(([Unlabel_data_O,Unlabel_data_Sc]))
	sio.savemat('data/Data_NFS_Source.mat', {'Train_data': Train_data_S,'Train_Y': Train_data_Y,'Test_data':Test_data_S,'Unlabel_data':Unlabel_data_S})
	
	