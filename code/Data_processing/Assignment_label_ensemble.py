#coding=utf-8
__author__ = "Hai Wang"
from sklearn.ensemble import RandomForestClassifier
import xgboost
import random
from sklearn.linear_model import LogisticRegression
import scipy.io as sio 
import numpy as np
import heapq
from sklearn.metrics import roc_auc_score
def fun_vstack(p,pre):
	if len(p)==0:
		p = pre
	else:
		p = np.vstack((p,pre))
	return p
def Assignment_label():
	Data_pre = sio.loadmat('data/Data.mat')
	Train_data = Data_pre['Train_data']
	Train_label = Data_pre['Train_Y']
	Unlabel_data = Data_pre['Unlabel_data']
	Test_data = Data_pre['Test_data']

	Train_label[Train_label==-1]=0
	Train_label = Train_label.ravel()
	iter = 10
	sample_iter = 10
	muti = [i for i in np.arange(1.0,2.5,0.2)]

	N = Train_data.shape[0]
	Validation_number = int(N*0.2)
	indices = np.random.permutation(N)

	Cur_Train_data = Train_data[indices[:Validation_number],:]
	Cur_Train_label = Train_label[indices[:Validation_number]]

	Cur_validation = Train_data[indices[Validation_number:],:]
	Cur_validation_label = Train_label[indices[Validation_number:]]


	clf1 = LogisticRegression(class_weight={1: 1, 0: 2.5},penalty='l2',C=0.15)
	clf2 = xgboost.XGBClassifier(n_estimators=600, learning_rate  = 0.1,colsample_bytree= 0.9, subsample= 0.7,objective='binary:logistic', max_delta_step= 0.8, max_depth=3, scale_pos_weight=0.4)
	clf3 = RandomForestClassifier(n_estimators=400, max_depth=12, max_features=0.9,n_jobs=-1)
	AUC = 0

	feature_number = Cur_Train_data.shape[1]
	feature_idx = list([x for x in range(feature_number)])
	for i in range(iter):
		pre1 = np.array([])
		label1 = np.array([])
		pre2 = np.array([])
		label2 = np.array([])
		pre3 = np.array([])
		label3 = np.array([])
		clf1.fit(Cur_Train_data, Cur_Train_label)
		label_VA = clf1.predict_proba(Cur_validation)[:,1]
		Cur_AUC = roc_auc_score(Cur_validation_label, label_VA)
		print(Cur_AUC)
		if Cur_AUC>AUC:
			AUC = Cur_AUC
		else:
			break
		for j in range(sample_iter):
			print('--- Fitting For iters #{0} / {1} ---'.format(j+1, sample_iter))
			Neg_label = np.where(Cur_Train_label==0)[0]
			Pos_label = np.where(Cur_Train_label==1)[0]
			Cal_Neg_number = int(len(Neg_label)*0.9)
			Neg_slice = random.sample(list(Neg_label), Cal_Neg_number)
			Idx_muti = random.sample(list(muti), 1)
			Pos_slice = random.sample(list(Pos_label), int(Cal_Neg_number*Idx_muti[0]))
			Index = np.array(Neg_slice+Pos_slice)
			S_Train_data = Cur_Train_data[Index,:] 
			S_y_train = Cur_Train_label[Index]
			feature_slice = random.sample(feature_idx, int(feature_number*0.8))
			ALL_Train_data = S_Train_data[:,feature_slice]

			print('model 1...')
			clf1.fit(ALL_Train_data, S_y_train)
			pre = clf1.predict_proba(Unlabel_data[:,feature_slice])[:,1]
			label = clf1.predict(Unlabel_data[:,feature_slice])
			pre1 = fun_vstack(pre1,pre)
			label1 = fun_vstack(label1,label)
			print('model 2...')
			clf2.fit(ALL_Train_data, S_y_train)
			pre = clf2.predict_proba(Unlabel_data[:,feature_slice])[:,1]
			label = clf2.predict(Unlabel_data[:,feature_slice])
			pre2 = fun_vstack(pre2,pre)
			label2 = fun_vstack(label2,label)
			print('model 3...')
			clf3.fit(ALL_Train_data, S_y_train)
			pre = clf3.predict_proba(Unlabel_data[:,feature_slice])[:,1]
			label = clf3.predict(Unlabel_data[:,feature_slice])
			pre3 = fun_vstack(pre3,pre)
			label3 = fun_vstack(label3,label)

		prediction = np.vstack((pre1,pre2,pre3))
		pre_mean = np.mean(prediction, axis=0)
		idx_pos = heapq.nlargest(100, range(len(pre_mean)), pre_mean.__getitem__)
		pre_mean = -1*pre_mean
		idx_neg = heapq.nlargest(100, range(len(pre_mean)), pre_mean.__getitem__)
		idx = np.array(idx_neg)

		label_pre = np.vstack((label1,label2,label3))
		label_p = np.sum(label_pre, axis=0)
		label_p = label_p[idx]

		label_p[label_p<int(sample_iter*3/2)] = 0
		label_p[label_p>=int(sample_iter*3/2)] = 1	
		right_idx = np.where(label_p==0)[0]
		label_assig = label_p[right_idx]
		data_assig = Unlabel_data[idx[right_idx],:]
		Cur_Train_data = np.vstack((Cur_Train_data,data_assig))
		Cur_Train_label = np.hstack((Cur_Train_label,label_assig))
		Unlabel_data = np.delete(Unlabel_data, idx[right_idx], 0)

	Cur_Train_data = np.vstack((Cur_Train_data,Cur_validation))
	Cur_Train_label = np.hstack((Cur_Train_label,Cur_validation_label))
	Cur_Train_label[Cur_Train_label==0] = -1
	sio.savemat('data/Data_SSL.mat', {'Train_data': Cur_Train_data,'Train_Y': Cur_Train_label,'Test_data':Test_data})	