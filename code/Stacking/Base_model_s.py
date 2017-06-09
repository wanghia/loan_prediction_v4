#coding=utf-8
__author__ = "Hai Wang"

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
base_models_s = [
        LinearSVC(C = 0.08,penalty = 'l2', class_weight='balanced', max_iter=1500, fit_intercept=False),
        LinearSVC(C = 0.02,penalty = 'l2', class_weight='balanced', max_iter=1500, fit_intercept=False),
        LogisticRegression(class_weight='balanced',penalty='l1',C=0.01,n_jobs=-1),
        LogisticRegression(class_weight='balanced',penalty='l2',C=0.15,n_jobs=-1),
        MLPClassifier(hidden_layer_sizes = 150, activation = 'relu',solver='sgd'),
        MLPClassifier(hidden_layer_sizes = 100, activation = 'logistic',solver='lbfgs'),
    ]
