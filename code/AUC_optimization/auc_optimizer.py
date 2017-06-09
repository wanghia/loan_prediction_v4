#coding=utf-8
'''
Created on Match 3, 2017
@author: weit
'''

import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
#from dataGeneration import generate_data
import auc_and_rank


def auc_optimizer(X, Y, max_pair_samples = 100000):
    '''
        X: feature matrix, each row is an instance
        Y: label vector, each row is a label
        max_pair_samples: # of pairs sampled
    '''

    X = preprocessing.scale(X, with_mean=False) # unit variance


    (X_new, Y_new) = auc_and_rank.createFeaturesForAUC(X, Y, max_pair_samples)

    #===================
    # since scipy does not like that all elements in the classifier are all positive we shall
    # make a workaround, we randomly flip half the samples and also their labels
    #===================
    [n_new, d] = X_new.shape
    rand_int = np.random.randint(n_new, size=int(n_new/3))
    X_new[rand_int,:] = -1 * X_new[rand_int,:]
    Y_new[rand_int,:] = -1 * Y_new[rand_int,:]

    # choose appropriate 'C' with cross validation
    C_val = np.arange(0.01, 0.5, 0.05)
    parameters = {'C': C_val}

    svr = LinearSVC(class_weight='balanced', max_iter=1000, fit_intercept=False)
    grid_search = GridSearchCV(svr, parameters, n_jobs = -1, scoring = 'roc_auc')

    grid_search.fit(X_new, Y_new.ravel())

    clf = grid_search.best_estimator_

    w = clf.coef_

    return w
