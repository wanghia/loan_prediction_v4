#coding=utf-8
__author__ = "Hai Wang"
from sklearn.linear_model import LogisticRegression
import scipy.io as sio 
import numpy as np
import pickle

def LR(Train_data,Train_label,Test_data,fname = ''):

	pkl_filename  = "model/model_LR.pkl"
	f = open(pkl_filename, 'wb')

	tmp_model_data = {'model': {}, 'selected_feature': {}}


	print('training...')
	# clf = LogisticRegression(class_weight={1: 1, -1: 2.5},penalty='l1',C=0.15)
	clf = LogisticRegression(class_weight={1: 1, 0: 2.5},penalty='l2',C=0.15) 
	clf.fit(Train_data, Train_label.ravel())
	
	feature_slice = list([i for i in range(Train_data.shape[1])])
	# save selected features
	tmp_model_data['selected_feature'][0] = feature_slice
	# save model 
	tmp_model_data['model'][0] = clf

	# write data to file
	pickle.dump(tmp_model_data, f, pickle.HIGHEST_PROTOCOL)

	f.close()
	print('test...')
	Test_pro = clf.predict_proba(Test_data)
	np.savetxt('prediction/' + fname + 'LR_prediction.csv', Test_pro[:,-1], fmt='%.3f', delimiter = ',') 
	print("Prediction results stored in the 'predction/" + fname + "LR_prediction.csv'")
