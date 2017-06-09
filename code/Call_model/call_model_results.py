#coding=utf-8
'''
	by weit, May 6, 2017
'''
import numpy as np
import pickle
def call_model_results(Test_data,f,K):
	S_test = np.zeros((Test_data.shape[0], K))
	print('number of models: ' + str(K))
	for i in range(K):
		# load the i-th model
		tmp_model_data = pickle.load(f)
		print('model ' + str(i+1) + '....')
		# print(tmp_model_data)
		S_test_i = np.zeros((Test_data.shape[0], len(tmp_model_data['model'])))
		# print(tmp_model_data)
		for j in range(len(tmp_model_data['model'])):

			# load the j-th selected features
			feature_slice = tmp_model_data['selected_feature'][j]

			# load the i-th model fed with the j-th fold features
			clf = tmp_model_data['model'][j]
			y_pred = clf.predict_proba(Test_data[:, feature_slice])[:, 1]
			S_test_i[:, j] = y_pred
		# print(S_test_i)
		S_test[:, i] = S_test_i.mean(1)
	return S_test