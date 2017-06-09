#coding=utf-8
__author__ = "Hai Wang"
import numpy as np
import auc_and_rank
from auc_optimizer import auc_optimizer


def AUC2(Train_data,Train_Y,Test_data):
    max_pair_samples=60000
    w = auc_optimizer(Train_data, Train_Y)

    predictions = auc_and_rank.predict_score_for_auc(Test_data, w)

    return predictions
