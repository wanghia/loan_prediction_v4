#coding=utf-8
__author__ = "Hai Wang"
#file name
Train_filename = 'data/Train_data.csv'
Train_Y_filename = 'data/Train_Y.csv'
Test_filename = 'data/Test_data.csv'
Train_write_file = 'data/Train_labele_data.csv'
Train_write_file_Y = 'data/Train_labele_data_Y.csv'
Unlabel_data_filename = 'data/unlabel_data.csv'

threshold_path = 'model/threshold.txt'
Location_path = 'model/Location_feature.txt'
Source_path = 'model/Source_feature.txt'
Tree_path = 'model/xgbfeature.txt'
group_path = 'model/xgbgroup.txt'

#Variable
Categorical_features = list([7,13,29,33,39,49,63,77,85,89,99,117,122,128,158])
binary_features = list([0,4,9,11,14,20,36,38,43,45,46,55,58,61,64,68,78,81,83,84,86,87,
91,92,93,96,97,98,104,105,107,108,109,112,113,119,121,123,124,125,126,127,129,130,131,132,
149,150,151,152])

Numerical = list([1,2,3,5,6,8,10,12,15,16,17,18,19,21,22,23,24,25,26,27,28,30,31,32,34,35,37,40,41,42,44,47,48,50,51,52,53,54,56,57,59,60,62,65,66,67,69,
	70,71,72,73,74,75,76,79,80,82,88,90,94,95,100,101,102,103,106,110,111,114,115,116,118,120,153,154,155,156,157,159])+list([x for x in range(133,149)])
Province_code = list([29])
Are_code = list([39])
City_code = list([13])
Location_code = Province_code+Are_code+City_code

log_idx = list([1,2,3,6,8,12,16,17,18,21,22,25,27,28,30,31,34,
	35,37,40,48,51,56,57,65,66,67,69,70,73,74,75,76,80,82,88,100,101,102,
	106,110,111,114,115,118,153,154,156,157,159,133,137,138,141,142,143,146,147,148])