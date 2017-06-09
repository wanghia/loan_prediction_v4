#coding=utf-8
__author__ = "Hai Wang"
import sys,os,time
import numpy as np
import random
import pickle
import copy
import scipy.io as sio
from sklearn.model_selection import KFold


class Ensemble_s(object):
	def __init__(self, LinearSVC_n, sampling ,n_folds, base_models):
		self.LinearSVC_n = LinearSVC_n
		self.Sample_iter = sampling
		self.n_folds = n_folds
		self.base_models = base_models
		
	def predict_score_for_auc(self,X, w):
		prediction_scores = X.dot( w.transpose())
		return prediction_scores.ravel()

	def fun_hstack(self, A, B):
		if len(A)==0:
			A = B
		else:
			A = np.hstack((A,B))
		return A

	def fit_predict(self, X, y, T):
		start_time = time.time()
		X = np.array(X)
		y = np.array(y)
		T = np.array(T)

		S_train = np.zeros((X.shape[0], len(self.base_models)))
		S_test = np.zeros((T.shape[0], len(self.base_models)))

		A_train = np.array([])
		A_test = np.array([])

		delete_id = []
		feature_number = X.shape[1]
		feature_idx = list([i for i in range(feature_number)])


		# save models and selected features into files,
		# which will be used in prediction process
		pkl_filename  = "model/models_stacking_linear.pkl"
		f = open(pkl_filename, 'wb')

		muti = [i for i in np.arange(1.0,2.5,0.2)]
		feature_sample = [i for i in np.arange(0.6,0.95,0.02)]


		for i, clf in enumerate(self.base_models):
			id_x = 0
			T_test = np.zeros((T.shape[0], self.Sample_iter))
			T_train = np.zeros((X.shape[0], self.Sample_iter))
			print('Fitting For Base Model #{0} / {1} ---'.format(i+1, len(self.base_models)))
			kf = KFold(n_splits = self.n_folds,shuffle = True)
			folds = list(kf.split(X))
			for k in range(self.Sample_iter):
				KFCV_test = np.zeros((T.shape[0], self.n_folds))
				print('--- Fitting For iters #{0} / {1} ---'.format(k+1, self.Sample_iter))
				tmp_model_data = {}
				tmp_model_data = {'model': {}, 'selected_feature': {}}
				for j, (train_idx, test_idx) in enumerate(folds):
					print('------ Fitting For Fold #{0} / {1} ---'.format(j+1, self.n_folds))
					Idx_f = random.sample(list(feature_sample), 1)
					feature_slice = random.sample(feature_idx, int(feature_number*Idx_f[0]))
					y_1 = y[train_idx]
					X_1 = X[train_idx]
					Neg_label = np.where(y_1==0)[0]
					Pos_label = np.where(y_1==1)[0]
					Cal_Neg_number = int(len(Neg_label)*0.9)
					Neg_slice = random.sample(list(Neg_label), Cal_Neg_number)
					Idx_muti = random.sample(list(muti), 1)
					Pos_slice = random.sample(list(Pos_label), int(Cal_Neg_number*Idx_muti[0]))
					Index = np.array(Neg_slice+Pos_slice)
					Train_data = X_1[Index,:] 
					y_train = y_1[Index]

					X_train = Train_data[:,feature_slice]

					X_holdout1 = X[test_idx]
					X_holdout = X_holdout1[:,feature_slice]

					clf.fit(X_train, y_train)
					if i>=self.LinearSVC_n:					
						y_pred = clf.predict_proba(X_holdout)[:,1]
						S_P = clf.predict_proba(T[:,feature_slice])[:,1]
					else:
						w = clf.coef_
						y_pred = self.predict_score_for_auc(X_holdout, w)
						S_P = self.predict_score_for_auc(T[:,feature_slice],w)
					T_train[test_idx, k] = y_pred
					KFCV_test[:, j] = S_P
					print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
					
					# save the j-th selected features
					tmp_model_data['selected_feature'][j] = copy.deepcopy(feature_slice)
					# save the i-th model fed with the j-th fold features
					tmp_model_data['model'][j] = copy.deepcopy(clf)
					#print(clf)


				pickle.dump(tmp_model_data, f, pickle.HIGHEST_PROTOCOL)
				T_test[:,k] = KFCV_test.mean(1)
			A_train = self.fun_hstack(A_train,T_train)
			A_test = self.fun_hstack(A_test,T_test)	
			if (i+1)%2==0:
				sio.savemat('model/Stacking_data_linear.mat', {'Train_data': A_train,'Train_Y':y,'Test_data':A_test})
		f.close()
		sio.savemat('model/Stacking_data_linear.mat', {'Train_data': A_train,'Train_Y':y,'Test_data':A_test})
		