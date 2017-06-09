#coding=utf-8
__author__ = "Hai Wang"
import xgboost
from sklearn.model_selection import GridSearchCV
import scipy.io as sio 
import numpy as np
import pickle
import warnings

def xgb_Feature_selection(file_name, save_file_name):
	warnings.filterwarnings('ignore')
	Data = sio.loadmat('data/Data_NFS_'+file_name+'.mat')
	Train_data = Data['Train_data']
	Train_label = Data['Train_Y']
	Test_data = Data['Test_data']
	Unlabel_data = Data['Unlabel_data']

	print('training...')
	clf = xgboost.XGBClassifier(n_estimators=800, learning_rate  = 0.1, colsample_bytree= 0.7, subsample= 0.7,objective='binary:logistic', max_delta_step= 0.8, max_depth=2, scale_pos_weight=0.8)

	clf.fit(Train_data, Train_label.ravel())
	
	coe = clf.feature_importances_
	# print(np.max(coe))
	# print(coe)
	index = np.argsort(-coe)

	import_index_len = len(np.where(coe!=0)[0])

	Save_list = list(index[:import_index_len])
	print(coe[Save_list])
	file_open = open(save_file_name,'wb')
	pickle.dump(Save_list,file_open)
	file_open.close()


	Train_data = Train_data[:,index[:import_index_len]]
	Test_data = Test_data[:,index[:import_index_len]]
	if len(Unlabel_data)>0:
		Unlabel_data = Unlabel_data[:,index[:import_index_len]]

	print('Number of features is ' + str(Train_data.shape[1]))
	sio.savemat('data/Data_'+file_name+'.mat', {'Train_data': Train_data,'Train_Y': Train_label,'Test_data':Test_data,'Unlabel_data':Unlabel_data})
if __name__=='__main__':
	xgb_Feature_selection('model/xgbfeature.txt')
