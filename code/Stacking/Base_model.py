#coding=utf-8
__author__ = "Hai Wang"
from Stacking import *
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier

base_models = [
        xgboost.XGBClassifier(
            n_estimators=800, learning_rate  = 0.1, colsample_bytree= 0.7, subsample= 0.7,objective='binary:logistic', max_delta_step= 0.8, max_depth=2, scale_pos_weight=0.8
        ),
        xgboost.XGBClassifier(
            n_estimators=900, learning_rate  = 0.02,colsample_bytree= 0.8, subsample= 0.8,objective='rank:pairwise', max_delta_step= 0.8, max_depth=3, scale_pos_weight=0.4
        ),
        xgboost.XGBClassifier(
            n_estimators=600, learning_rate  = 0.1,colsample_bytree= 0.9, subsample= 0.9,objective='binary:logistic', max_delta_step= 0.8, max_depth=4, scale_pos_weight=0.6
        ),
        xgboost.XGBClassifier(
            n_estimators=800, learning_rate  = 0.05,colsample_bytree= 0.9, subsample= 0.8,objective='binary:logistic', max_delta_step= 0.8, max_depth=2, scale_pos_weight=0.8
        ),
        RandomForestClassifier(n_estimators=500, max_depth=10, max_features=0.9,n_jobs = -1,criterion='gini'),
        RandomForestClassifier(n_estimators=600, max_depth=10, max_features=0.8,n_jobs = -1, criterion='entropy'),
        GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=4, subsample=0.7, min_samples_split=7
        ),
        GradientBoostingClassifier(
        n_estimators=700, learning_rate=0.05, max_depth=2, subsample=0.9, min_samples_split=8
        ),
        AdaBoostClassifier(n_estimators=500,learning_rate=0.1),
        AdaBoostClassifier(n_estimators=600,learning_rate=0.05),
   ]
