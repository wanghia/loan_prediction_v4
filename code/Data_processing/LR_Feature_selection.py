#coding=utf-8
__author__ = "Hai Wang"
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import scipy.io as sio 
import numpy as np
import pickle
import warnings

def LR_Feature_selection(fname,save_file_name):
	warnings.filterwarnings('ignore')
	Data = sio.loadmat('data/Data_NFS'+fname+'.mat')
	Train_data = Data['Train_data']
	Train_label = Data['Train_Y']
	Test_data = Data['Test_data']
	Unlabel_data = Data['Unlabel_data']


	print('GridSearchCV...')
	clf = LogisticRegression(class_weight='balanced',penalty='l1',C=1)
	# param_test = {'class_weight':[{-1:2, 1:1}],'penalty':['l1'], 'C':[0.15]}
	param_test = {'class_weight':[{-1:2, 1:1},{-1:1.5, 1:1}],'penalty':['l1'], 'C':[0.15,0.2,0.25]}
	gsearch = GridSearchCV(clf, param_grid = param_test, scoring='roc_auc',n_jobs=1, cv=2)
	gsearch.fit(Train_data, Train_label.ravel())
	best_parameters, score, _ = max(gsearch.grid_scores_, key=lambda x: x[1])

	# print('Best score: ' + str(score))
	print('best_parameters:')
	print(best_parameters)
	print('training...')
	clf = LogisticRegression(class_weight=best_parameters['class_weight'],penalty = best_parameters['penalty'], C=best_parameters['C']) 
	clf.fit(Train_data, Train_label.ravel())
	coe = abs(clf.coef_[0])
	index = np.argsort(-coe)

	import_index_len = len(np.where(coe!=0)[0])

	Save_list = list(index[:import_index_len])
	file_open = open(save_file_name,'wb')
	pickle.dump(Save_list,file_open)
	file_open.close()


	Train_data = Train_data[:,index[:import_index_len]]
	Test_data = Test_data[:,index[:import_index_len]]
	if len(Unlabel_data)>0:
		Unlabel_data = Unlabel_data[:,index[:import_index_len]]

	print('Number of features is ' + str(Train_data.shape[1]))
	sio.savemat('data/Data'+fname+'.mat', {'Train_data': Train_data,'Train_Y': Train_label,'Test_data':Test_data,'Unlabel_data':Unlabel_data})

