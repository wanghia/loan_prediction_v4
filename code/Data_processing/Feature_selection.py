#coding=utf-8
__author__ = "Hai Wang"

import scipy.io as sio
import os
import pickle
import numpy as np

def Feature_selection(fname,save_file_name):
	Data = sio.loadmat('data/Data_NFS'+fname+'.mat')
	Train_data = Data['Train_data']
	Train_label = Data['Train_Y']
	Test_data = Data['Test_data']
	Unlabel_data = Data['Unlabel_data']

	index = []
	if os.path.getsize(save_file_name):
		with open(save_file_name,'rb') as file_open:
			unpickler = pickle.Unpickler(file_open)
			index = unpickler.load()

	Train_data = Train_data[:,index]
	Test_data = Test_data[:,index]
	if len(Unlabel_data)>0:
		Unlabel_data = Unlabel_data[:,index]

	print('Number of features is ' + str(Train_data.shape[1]))
	sio.savemat('data/Data'+fname+'.mat', {'Train_data': Train_data,'Train_Y': Train_label,'Test_data':Test_data,'Unlabel_data':Unlabel_data})

