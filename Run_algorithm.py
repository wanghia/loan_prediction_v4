#coding=utf-8
__author__ = "Hai Wang"
import sys,os,time
sys.path.append('code/Learning_algorithm')
sys.path.append('code/Data_processing')
sys.path.append('code/Stacking')
sys.path.append('code/AUC_optimization')
from Assignment_label_ensemble import *
from  logistic_regression import *
from  Xgboost import *
from Preprocessing import *
from Stacking_s import *
from Base_model_s import *
from Stacking import *
from Base_model import *
from AUC1 import *
from AUC2 import *



if __name__ == '__main__':


	start_time = time.time()
	# Data preprocessing and Feature Selection
	print('Data preprocessing and Feature Selection...')
	preprocess()
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	
	Data = sio.loadmat('data/Data_linear.mat')
	Train_data = Data['Train_data']
	Train_label = Data['Train_Y']
	Test_data = Data['Test_data']
	Train_label[Train_label==-1]=0


	#Base on logistic regression
	print('Train a logistic regression model ...')
	LR(Train_data,Train_label,Test_data,fname = 'SSL_')
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	#Base on stacking1
	print('stacking1...')
	ensemble_s = Ensemble_s(
		LinearSVC_n = 2,
		sampling = 10,
		n_folds=6,
		base_models=base_models_s
	)
	ensemble_s.fit_predict(X=Train_data, y=Train_label.ravel(), T=Test_data)
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))


	Data = sio.loadmat('data/Data_tree.mat')
	Train_data = Data['Train_data']
	Train_label = Data['Train_Y']
	Test_data = Data['Test_data']
	Train_label[Train_label==-1]=0

	#Base on xgboost
	print('Train a xgboost  model...')
	ex_xgb(Train_data,Train_label,Test_data,fname = 'SSL_')
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	#Base on stacking
	print('stacking...')
	ensemble = Ensemble(
		n_folds=6,
		sampling = 10,
		base_models=base_models
	)
	ensemble.fit_predict(X=Train_data, y=Train_label.ravel(), T=Test_data)
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	

	#load stacking results
	print('load stacking results...')
	Data = sio.loadmat('model/Stacking_data_linear.mat')
	Train_data_l = Data['Train_data']
	Train_label = Data['Train_Y']
	Test_data_l = Data['Test_data']
	Train_label[Train_label==-1]=0
	 
	#load stacking results
	print('load stacking results...')
	Data = sio.loadmat('model/Stacking_data_tree.mat')
	Train_data_t = Data['Train_data']
	Test_data_t = Data['Test_data']

	Train_data = np.hstack((Train_data_l,Train_data_t))
	Test_data = np.hstack((Test_data_l,Test_data_t))
	print(Train_data.shape)
	print(Test_data.shape)


	
	#AUC optimization1
	print('AUC optimization1...')
	y_pred = AUC1(Train_data,Train_label,Test_data)
	np.savetxt('prediction/stacking_AUC_prediction1.csv', y_pred, fmt='%.3f', delimiter = ',') 
	print("Prediction results stored in the 'predction/stacking_AUC_prediction1.csv'")
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	
	#AUC optimization2
	print('AUC optimization2')
	y_pred = AUC2(Train_data,Train_label,Test_data)
	np.savetxt('prediction/stacking_AUC_prediction2.csv', y_pred, fmt='%.3f', delimiter = ',') 
	print("Prediction results stored in the 'predction/stacking_AUC_prediction2.csv'")
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	


	
