#coding=utf-8
__author__ = "Hai Wang"
import sys,os,time
import pickle
sys.path.append('code/Learning_algorithm')
sys.path.append('code/Data_processing')
sys.path.append('code/AUC_optimization')
sys.path.append('code/Call_model')
sys.path.append('code/Stacking')

from Preprocessing_Finsh import *
from call_model_results_linear import *
from call_model_results import *
from AUC1 import *
from AUC2 import *
from Base_model import *
from Base_model_s import *
if __name__ == '__main__':

	start_time = time.time()
	#Data preprocessing and Feature Selection
	print('Data preprocessing and Feature Selection...')
	Preprocessing_Finsh()
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))


	#load Test data
	Data = sio.loadmat('data/Data_linear.mat')
	Test_data = Data['Test_data']

	print('Predicting using logistic regression model...')
	with open('model/model_LR.pkl', 'rb') as f:
		LR_prediction = call_model_results(Test_data,f,1)
	np.savetxt('prediction/SSL_LR_prediction.csv', LR_prediction, fmt='%.3f', delimiter = ',') 
	print("Prediction results stored in the 'predction/SSL_LR_prediction.csv'")
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))


	print('obtain predicted results of test data...')
	with open('model/models_stacking_linear.pkl', 'rb') as f:
		stacking_feature_linear = call_model_results_linear(Test_data,f,len(base_models_s)*10,2*10)
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
	sio.savemat('model/Stacking_prediction_linear.mat', {'Test_data':stacking_feature_linear})

	#load Test data
	Data = sio.loadmat('data/Data_tree.mat')
	Test_data = Data['Test_data']
	print('Predicting using xgboost model...')
	with open('model/model_xgb.pkl', 'rb') as f:
		xgb_prediction = call_model_results(Test_data,f,1)
	np.savetxt('prediction/SSL_xgb_prediction.csv', xgb_prediction, fmt='%.3f', delimiter = ',') 
	print("Prediction results stored in the 'predction/SSL_xgb_prediction.csv'")
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	print('obtain predicted results of test data...')
	with open('model/models_stacking.pkl', 'rb') as f:
		stacking_feature_tree = call_model_results(Test_data,f,len(base_models)*10)
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
	sio.savemat('model/Stacking_prediction_tree.mat', {'Test_data':stacking_feature_tree})

	#load stacking results
	print('load stacking results...')
	Data = sio.loadmat('model/Stacking_data_linear.mat')
	Train_data_l = Data['Train_data']
	Train_label = Data['Train_Y']
	Train_label[Train_label==-1]=0
	 
	#load stacking results
	print('load stacking results...')
	Data = sio.loadmat('model/Stacking_data_tree.mat')
	Train_data_t = Data['Train_data']

	Train_data = np.hstack((Train_data_l,Train_data_t))
	print(Train_data.shape)

	Data = sio.loadmat('model/Stacking_prediction_tree.mat')
	stacking_feature_tree = Data['Test_data']
	Data = sio.loadmat('model/Stacking_prediction_linear.mat')
	stacking_feature_linear = Data['Test_data']
	stacking_feature = np.hstack((stacking_feature_linear,stacking_feature_tree))

	#AUC optimization1
	print('AUC optimization1...')
	y_pred = AUC1(Train_data,Train_label,stacking_feature)
	np.savetxt('prediction/stacking_AUC_prediction1.csv', y_pred, fmt='%.5f', delimiter = ',') 
	print("Prediction results stored in the 'predction/stacking_AUC_prediction1.csv'")
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	
	#AUC optimization2
	print('AUC optimization2')
	y_pred = AUC2(Train_data,Train_label,stacking_feature)
	np.savetxt('prediction/stacking_AUC_prediction2.csv', y_pred, fmt='%.5f', delimiter = ',') 
	print("Prediction results stored in the 'predction/stacking_AUC_prediction2.csv'")
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	

	
	
	





	
