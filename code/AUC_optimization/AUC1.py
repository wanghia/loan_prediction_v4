#coding=utf-8
__author__ = "Hai Wang"

import numpy as np
from sklearn.linear_model import LogisticRegression
import scipy.io as sio
from sklearn.model_selection import GridSearchCV
import warnings


def AUC1(Train_data,Train_Y,Test_data):
	# warnings.filterwarnings('ignore')
	# clf = LogisticRegression(class_weight='balanced',penalty='l1',C=1)

	# param_test = {'class_weight':[{0:1, 1:1},{0:2, 1:1},'balanced'],'penalty':['l2'], 'C':[0.1,0.15,0.2]}
	# gsearch = GridSearchCV(clf, param_grid = param_test, scoring='roc_auc',n_jobs=-1, cv=3)
	# gsearch.fit(Train_data, Train_Y.ravel())
	# best_parameters, score, _ = max(gsearch.grid_scores_, key=lambda x: x[1])


	print('training...')
	clf = LogisticRegression(class_weight={0: 1, 1: 1},penalty = 'l1', C=0.01)
	clf.fit(Train_data, Train_Y.ravel())
	y_pred = clf.predict_proba(Test_data)
	return y_pred[:,1]
