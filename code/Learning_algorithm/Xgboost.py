#coding=utf-8
__author__ = "Hai Wang"
import xgboost
import scipy.io as sio 
import numpy as np
import pandas as pd 
import pickle
def ex_xgb(Train_data,Train_label,Test_data,fname = ''):

	pkl_filename  = "model/model_xgb.pkl"
	f = open(pkl_filename, 'wb')

	tmp_model_data = {'model': {}, 'selected_feature': {}}

	Train_label[Train_label==-1]=0
	print('training...')


	# clf = xgboost.XGBClassifier(n_estimators=800, learning_rate  = 0.05,colsample_bytree= 0.9, subsample= 0.7,objective='binary:logistic', max_delta_step= 0.8, max_depth=3, scale_pos_weight=0.4)

	clf = xgboost.XGBClassifier(n_estimators=800, learning_rate  = 0.1, colsample_bytree= 0.7, subsample= 0.7,objective='binary:logistic', max_delta_step= 0.8, max_depth=2, scale_pos_weight=0.8)

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
	Test_pro = clf.predict_proba(Test_data)[:,1]
	np.savetxt('prediction/' + fname + 'xgb_prediction.csv', Test_pro, fmt='%.3f', delimiter = ',') 
	print("Prediction results stored in the 'predction/"+ fname +"xgb_prediction.csv'")