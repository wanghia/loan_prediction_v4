#coding=utf-8
__author__ = "Hai Wang"
import os
import Variable_file
import numpy as np
import copy
import random
import pickle
import pandas as pd
from sklearn import preprocessing
from Entropy import *
import warnings
import scipy.io as sio
from scipy.stats import mode

def Preprocessing_v(threshold_path):
	#The entry of data processing 
	try:
		Train_read = open(Variable_file.Train_filename, 'r')
	except IOError:
		print ('Can not open', Variable_file.Train_filename)
		return 
	else:
		try:
			Train_Y_read = open(Variable_file.Train_Y_filename, 'r')
		except IOError:
			print ('Can not open', Variable_file.Train_Y_filename)
			return 
		else:
			#Delete unlabeled data in training set
			Del_unlabeled_data(Train_read,Train_Y_read)
			Train_read.close()
			Train_Y_read.close()
	
	print('Delete the missing variables and encoding categorical features...')
	onehot_one_hot_encoding(threshold_path)

def onehot_one_hot_encoding(threshold_path):
	#Delete the missing variables and encoding categorical features
	Train_data_Y = np.loadtxt(open(Variable_file.Train_write_file_Y,"rb"),delimiter=",",skiprows=0)
	Train_data_matrix = np.loadtxt(open(Variable_file.Train_write_file,"rb"),delimiter=",",skiprows=0)
	Test_data_matrix = np.loadtxt(open(Variable_file.Test_filename,"rb"),delimiter=",",skiprows=0)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		Unlabel_data_matrix = np.loadtxt(open(Variable_file.Unlabel_data_filename,"rb"),delimiter=",",skiprows=0)
		
	Train_number_data = Train_data_matrix.shape[0]
	Test_number_data = Test_data_matrix.shape[0]
	
	if len(Unlabel_data_matrix)>0:
		Unlabel_data_matrix[:,310] = 0
		Unlabel_number_data = Unlabel_data_matrix.shape[0]
		Data = np.vstack((Train_data_matrix,Unlabel_data_matrix,Test_data_matrix))
	else:
		Data = np.vstack((Train_data_matrix,Test_data_matrix))
	All_number_data = Data.shape[0]
	Test_number_start = All_number_data - Test_number_data
	Col_number = Data.shape[1]
	Del_missing_variables = []
	Features_id = []
	for i in range(0,Col_number):
		if i%2==0:
			Features_id.append(i)
			Cur_col = Train_data_matrix[:,i]
			if sum(Cur_col) < 0.02 * Train_number_data:
				Del_missing_variables.append(int(i/2))

	Missing_data = Data[:,Features_id]
	Missing_feature = Missing_data.shape[1] - Missing_data.sum(axis=1)

	Processing_data = np.delete(Data,Features_id, 1)
	Train_data_pro = Processing_data[0:Train_number_data,:]
	Test_data_pro = Processing_data[Test_number_start:,:]
	Train_fit = np.vstack((Train_data_pro,Test_data_pro))
	df_train = pd.DataFrame(data = Train_fit)
	df = pd.DataFrame(data = Processing_data)
	print('groupby...')
	group_feature = group_by(df,df_train)
	print('Delete the missing variables:' + str(Del_missing_variables))
	

	print('Encoding categorical features...')
	enc = preprocessing.OneHotEncoder()

	Feature_len = Processing_data.shape[1]
	One_hot_data = np.array([])

	Unique_feature_value_idx = []
	#Save location feature
	Location_feature = np.array([])
	#Save Numerical feature
	Numerical_feature = np.array([])
	#sort feature
	Sort_feature = np.array([])
	scale_results_data = np.array([])
	Raw_results_data = np.array([])

	Dic_threshold = {}
	if os.path.getsize(threshold_path):
		with open(threshold_path,'rb') as file_threshold:
			unpickler = pickle.Unpickler(file_threshold)
			Dic_threshold = unpickler.load()

	Categorical_binary_features = Variable_file.Categorical_features + Variable_file.binary_features
	print('Encoding categorical features '+ str(Categorical_binary_features)+'...')
	(scale_results_data,Raw_results_data) = Combination_entry(Dic_threshold,Processing_data,Missing_data,Train_number_data,Train_data_Y,enc,Test_number_start)
	for i in range(0,Feature_len):
		Select_feature = Processing_data[:,i]
		Train_feature = Processing_data[:Train_number_data,i]
		Select_feature_flag = Missing_data[:,i] 
		Different_train_feature_value = np.unique(Train_feature)
		if len(Different_train_feature_value)==1:
			#Delete features which contain only one value
			Unique_feature_value_idx.append(i)
			continue
		elif i in Del_missing_variables:
			#Delete features which contain a large number of missing value
			continue 
		elif i in Variable_file.Location_code:
			Select_feature[Select_feature_flag==0] = -999
			One_hot_feature = one_hot_function(Select_feature,enc,Test_number_start)
			Location_feature = hstack_mx(Location_feature,One_hot_feature)

		elif i in Categorical_binary_features:
			#Encode Categorical and binary features
			#Deal with missing data
			Select_feature[Select_feature_flag==0] = -999
			One_hot_feature = one_hot_function(Select_feature,enc,Test_number_start)
			One_hot_data = hstack_mx(One_hot_data,One_hot_feature)
		elif i in Variable_file.Numerical:
			#Numerical to discrete
			(scale_feature_N,raw_feature_N) = Numerical_to_discrete_entropy(Dic_threshold,i,Select_feature,Select_feature_flag,Train_number_data,Train_data_Y,enc,Test_number_start)
			scale_results_data = np.hstack((scale_results_data,scale_feature_N))
			Raw_results_data = np.hstack((Raw_results_data,raw_feature_N))

			# Raw_feature = pd.DataFrame(data=Select_feature)
			# Raw_feature_rank = Raw_feature.rank(ascending = True, method = 'max')
			# results = np.array(Raw_feature_rank[0])
			# results = results.reshape(len(results),1)
			# Sort_feature = hstack_mx(Sort_feature,results)
		else:
			print('feature' + str(i) + '?')
	Select_missing_flag = np.array([1 for x in range(len(Missing_feature))])
	(scale_feature_N,raw_feature_N) = Numerical_to_discrete_entropy(Dic_threshold,-4,Missing_feature,Select_missing_flag,Train_number_data,Train_data_Y,enc,Test_number_start)
	scale_results_data = np.hstack((scale_results_data,scale_feature_N))
	Raw_results_data = np.hstack((Raw_results_data,raw_feature_N))
	
	Train_data_O = One_hot_data[0:Train_number_data,:]
	Unlabel_data_O = One_hot_data[Train_number_data:Test_number_start,:]
	Test_data_O = One_hot_data[Test_number_start:,:]

	Train_data_Ra = Raw_results_data[0:Train_number_data,:]
	Unlabel_data_Ra = Raw_results_data[Train_number_data:Test_number_start,:]
	Test_data_Ra = Raw_results_data[Test_number_start:,:]

	Train_data_Sc = scale_results_data[0:Train_number_data,:]
	Unlabel_data_Sc = scale_results_data[Train_number_data:Test_number_start,:]
	Test_data_Sc = scale_results_data[Test_number_start:,:]
	

	Train_data_L = Location_feature[0:Train_number_data,:]
	Unlabel_data_L = Location_feature[Train_number_data:Test_number_start,:]
	Test_data_L = Location_feature[Test_number_start:,:]

	# Train_data_So = Sort_feature[0:Train_number_data,:]
	# Unlabel_data_So = Sort_feature[Train_number_data:Test_number_start,:]
	# Test_data_So = Sort_feature[Test_number_start:,:]

	Train_data_group = group_feature[0:Train_number_data,:]
	Unlabel_data_group = group_feature[Train_number_data:Test_number_start,:]
	Test_data_group = group_feature[Test_number_start:,:]
	


	file_threshold = open('model/threshold.txt','wb')
	pickle.dump(Dic_threshold,file_threshold)
	file_threshold.close()

	print('One hot encoding is complete')
	print('The features Which contain only one value ' + str(Unique_feature_value_idx) + ' have been deleted')
	# sio.savemat('data/Data_NFS_Numerical_rank.mat', {'Train_data': Train_data_So,'Train_Y': Train_data_Y,'Test_data':Test_data_So,'Unlabel_data':Unlabel_data_So})
	sio.savemat('data/Data_NFS_One_hot.mat', {'Train_data': Train_data_O,'Train_Y': Train_data_Y,'Test_data':Test_data_O,'Unlabel_data':Unlabel_data_O})
	sio.savemat('data/Data_NFS_Location.mat', {'Train_data': Train_data_L,'Train_Y': Train_data_Y,'Test_data':Test_data_L,'Unlabel_data':Unlabel_data_L})
	sio.savemat('data/Data_NFS_scale.mat', {'Train_data': Train_data_Sc,'Train_Y': Train_data_Y,'Test_data':Test_data_Sc,'Unlabel_data':Unlabel_data_Sc})
	sio.savemat('data/Data_NFS_raw.mat', {'Train_data': Train_data_Ra,'Train_Y': Train_data_Y,'Test_data':Test_data_Ra,'Unlabel_data':Unlabel_data_Ra})
	sio.savemat('data/Data_NFS_group.mat', {'Train_data': Train_data_group,'Train_Y': Train_data_Y,'Test_data':Test_data_group,'Unlabel_data':Unlabel_data_group})
	
def group_by(df,df_train):
	# #13-29-39
	iddx = list([13,29,39])+list([1,2,3,5,6,8,10,12,15,16,17,18,19,21,22,23,24,25,26,27,28,30,31,32,34,35,37,40,41,42,44,47,48,50,51,52,53,54,56,57,59,60,62,65,66,67,69,
	70,71,72,73,74,75,76,79,80,82,88,90,94,95,100,101,102,103,106,110,111,114,115,116,118,120,153,154,155,156,157,159])+list([x for x in range(133,149)])
	cur_df = df.loc[:,iddx]
	cur_df_train = df_train.loc[:,iddx]
	groupby1 = cur_df_train.groupby([13,29,39]).mean()
	groupby1 = groupby1.reset_index()
	mean1 = cur_df.merge(groupby1,on = [13,29,39],how='left')
	mean1 = mean1.fillna(0)
	age_ret = np.array(mean1)
	ret = age_ret[:,len(iddx):]

	groupby2 = cur_df_train.groupby([13,29,39]).sum()
	groupby2 = groupby2.reset_index()
	sum1 = cur_df.merge(groupby2,on = [13,29,39],how='left')
	sum1 = sum1.fillna(0)
	age_ret = np.array(sum1)
	ret2 = age_ret[:,len(iddx):]
	ret = np.hstack((ret,ret2))


	iddx2 = list([2,14])+list([1,3,5,6,8,10,12,15,16,17,18,19,21,22,23,24,25,26,27,28,30,31,32,34,35,37,40,41,42,44,47,48,50,51,52,53,54,56,57,59,60,62,65,66,67,69,
	70,71,72,73,74,75,76,79,80,82,88,90,94,95,100,101,102,103,106,110,111,114,115,116,118,120,153,154,155,156,157,159])+list([x for x in range(133,149)])

	cur_df = df.loc[:,iddx2]
	cur_df_train = df_train.loc[:,iddx2]
	groupby3 = cur_df_train.groupby([2,14]).mean()
	groupby3 = groupby3.reset_index()
	mean3 = cur_df.merge(groupby3,on = [2,14],how='left')
	mean3 = mean3.fillna(0)
	age_ret = np.array(mean3)
	ret3 = age_ret[:,len(iddx2):]
	ret = np.hstack((ret,ret3))

	groupby4 = cur_df_train.groupby([2,14]).sum()
	groupby4 = groupby4.reset_index()
	sum4 = cur_df.merge(groupby4,on = [2,14],how='left')
	sum4 = sum4.fillna(0)
	age_ret = np.array(sum4)
	ret4 = age_ret[:,len(iddx2):]
	ret = np.hstack((ret,ret4))
	return ret
	
def Numerical_to_discrete_entropy(Dic_threshold,i,Select_feature,Select_feature_flag,Train_number_data,Train_data_Y,enc,Test_number_start):
	Train_Select_feature = Select_feature[:Train_number_data]
	Cur_feature_flag = Select_feature_flag[:Train_number_data]
	cur_f = Train_Select_feature[Cur_feature_flag==1]
	cur_label = Train_data_Y[Cur_feature_flag==1]
	Cur_data = np.vstack((cur_f,cur_label)).transpose()
	flag = 0
	Cat_feature = np.array([])
	if Cur_data.shape[0]>100 and Cur_data.shape[1]==2:
		if i in Dic_threshold:
			cut_list = Dic_threshold[i]
		else:
			Cur_feature = pd.DataFrame(data=Cur_data)
			result = Discretizer(Cur_feature, 0, 1)
			cut_list = result._cuts[0]
		if len(cut_list)>0:
			if i>=0:
				print('The best cut points of feature '+ str(i) + ' is ' + str(cut_list))
			Dic_threshold[i] = cut_list
			(Cat_feature,scale_feature) = Entropy_dis(i,cut_list,Select_feature,Select_feature_flag,enc,Test_number_start,Train_number_data)
			flag = 1
	if flag == 0:
		scale_feature = Deal_Scale(i,Select_feature,Select_feature_flag,Train_number_data,Test_number_start)
		scale_feature = scale_feature.reshape((len(scale_feature),1))
		# Not_miss_value = Select_feature[Select_feature_flag==1]
		# Not_miss_value = preprocessing.scale(Not_miss_value)
		# Select_feature[Select_feature_flag==1] = Not_miss_value
		scale_feature = np.hstack((Select_feature_flag.reshape((len(Select_feature_flag),1)),scale_feature))
	Select_feature = Select_feature.reshape((len(Select_feature),1))
	scale_feature_N = hstack_mx(Cat_feature,scale_feature)
	raw_feature_N = hstack_mx(Cat_feature,Select_feature)
	return (scale_feature_N,raw_feature_N)

def Deal_Scale(i,Select_feature,Select_feature_flag,Train_number_data,Test_number_start):
	if i>0 and i not in Variable_file.log_idx:
		Train_feature = Select_feature[:Train_number_data]
		Train_feature_flag = Select_feature_flag[:Train_number_data]
		Unlabel_feature = Select_feature[Train_number_data:Test_number_start]
		Unlabel_feature_flag = Select_feature_flag[Train_number_data:Test_number_start]
		Test_feature = Select_feature[Test_number_start:]
		Test_feature_flag = Select_feature_flag[Test_number_start:]

		Train_feature_final = copy.deepcopy(Train_feature)
		Train_Not_miss_value = Train_feature[Train_feature_flag==1]
		if len(Train_Not_miss_value)>0:
			Train_Not_miss_value = preprocessing.scale(Train_Not_miss_value)
			Train_feature_final[Train_feature_flag==1] = Train_Not_miss_value


		Test_feature_S = np.hstack((Train_feature,Test_feature))
		Test_feature_S_flag = np.hstack((Train_feature_flag,Test_feature_flag))


		Test_feature_final = copy.deepcopy(Test_feature_S)
		Test_Not_miss_value = Test_feature_S[Test_feature_S_flag==1]
		if len(Test_Not_miss_value)>0:
			Test_Not_miss_value = preprocessing.scale(Test_Not_miss_value)
			Test_feature_final[Test_feature_S_flag==1] = Test_Not_miss_value

		All_feature_final = copy.deepcopy(Select_feature)
		ALL_Not_miss_value = Select_feature[Select_feature_flag==1]
		if len(ALL_Not_miss_value)>0:
			ALL_Not_miss_value = preprocessing.scale(ALL_Not_miss_value)
			All_feature_final[Select_feature_flag==1] = ALL_Not_miss_value

		All_feature_final[Test_number_start:] = Test_feature_final[Train_number_data:]
		All_feature_final[:Train_number_data] = Train_feature_final
		return All_feature_final
	else:
		All_feature_final = copy.deepcopy(Select_feature)
		Not_miss_value = All_feature_final[Select_feature_flag==1]
		if len(Not_miss_value)>0:
			Not_miss_value = np.log1p(Not_miss_value)
			All_feature_final[Select_feature_flag==1] = Not_miss_value
		return All_feature_final




def Entropy_dis(i,cut_list,Select_feature,Select_feature_flag,enc,Test_number_start,Train_number_data):
	Cat_feature = np.array([-1 for j in range(len(Select_feature))])
	for j in range(len(cut_list)):
		if j==0:
			Cat_feature[Select_feature <= cut_list[j]] = j
		elif j==len(cut_list)-1:
			Cat_feature[Select_feature > cut_list[j]] = j
		else:
			Cat_feature[np.logical_and(Select_feature > cut_list[j-1], Select_feature <= cut_list[j])] = j
	Cat_feature[Select_feature_flag==0] = -1
	Cat_feature = one_hot_function(Cat_feature,enc,Test_number_start)

	scale_feature = Deal_Scale(i,Select_feature,Select_feature_flag,Train_number_data,Test_number_start)

	scale_feature = scale_feature.reshape((len(scale_feature),1))
	# Not_miss_value = Select_feature[Select_feature_flag==1]
	# Not_miss_value = preprocessing.scale(Not_miss_value)
	# Select_feature[Select_feature_flag==1] = Not_miss_value
	# All_feature = np.hstack((Cat_feature,Select_feature.reshape((len(Select_feature),1))))

	return (Cat_feature,scale_feature)


def Combination_entry(Dic_threshold,Processing_data,Missing_data,Train_number_data,Train_data_Y,enc,Test_number_start):
	Cur_results = np.array([])
	Raw_results = np.array([])
	# feature 156 / feature 155
	Combination_results = Combination_feature(Processing_data[:,156],Processing_data[:,155])
	(scale_feature_N,raw_feature_N) = Numerical_to_discrete_entropy(Dic_threshold,-1,Combination_results,Missing_data[:,155],Train_number_data,Train_data_Y,enc,Test_number_start)
	Cur_results = hstack_mx(Cur_results,scale_feature_N)
	Raw_results = hstack_mx(Raw_results,raw_feature_N)

	# feature 156 / feature 154
	Combination_results = Combination_feature(Processing_data[:,156],Processing_data[:,154])
	(scale_feature_N,raw_feature_N) = Numerical_to_discrete_entropy(Dic_threshold,-2,Combination_results,Missing_data[:,154],Train_number_data,Train_data_Y,enc,Test_number_start)
	Cur_results = hstack_mx(Cur_results,scale_feature_N)
	Raw_results = hstack_mx(Raw_results,raw_feature_N)

	# feature 102 / feature 100
	Combination_results = Combination_feature(Processing_data[:,102],Processing_data[:,100])
	(scale_feature_N,raw_feature_N) = Numerical_to_discrete_entropy(Dic_threshold,-3,Combination_results,Missing_data[:,100],Train_number_data,Train_data_Y,enc,Test_number_start)
	Cur_results = hstack_mx(Cur_results,scale_feature_N)
	Raw_results = hstack_mx(Raw_results,raw_feature_N)

	return (Cur_results,Raw_results)



def Combination_feature(Mol, Den):
	Den_f = copy.deepcopy(Den)
	Den_f[Den_f<=0] = -1
	results = Mol/Den_f
	results[Den_f==-1] = 0
	return results


def hstack_mx(One_hot_data,One_hot_feature):
	if len(One_hot_data)==0:
		One_hot_data = One_hot_feature
	else:
		One_hot_data = np.hstack((One_hot_data,One_hot_feature))
	return One_hot_data

def one_hot_function(Select_feature,enc,Test_number_start):
	Select_feature = Select_feature.max() - Select_feature
	Select_feature = Select_feature.reshape((len(Select_feature),1))
	Train_feature = Select_feature[:Test_number_start]
	Test_feature = Select_feature[Test_number_start:]
	Test_feature_unique = np.unique(Test_feature)
	rand_idx = mode(list(Train_feature))[0]
	for test_f in Test_feature_unique:
		if test_f not in Train_feature:
			Test_feature[Test_feature==test_f] = rand_idx[0]
	enc.fit(Train_feature)
	Train_feature = enc.transform(Train_feature).toarray()
	Test_feature = enc.transform(Test_feature).toarray()
	return np.vstack((Train_feature,Test_feature))


def Del_unlabeled_data(Train_read,Train_Y_read):
	# Delete the unlabelled data
	File_data_Y = open(Variable_file.Train_write_file_Y, 'w')
	File_data = open(Variable_file.Train_write_file, 'w')
	File_unlabel_data = open(Variable_file.Unlabel_data_filename, 'w')
	Train_data_Y = np.loadtxt(Train_Y_read,delimiter=",",skiprows=0)
	data_count = 0
	labeled_data_count = 0
	for line in Train_read:
		if Train_data_Y[data_count]!=0:
			labeled_data_count = labeled_data_count + 1
			File_data.write(line)
			File_data_Y.write(str(int(Train_data_Y[data_count]))+'\n')
		else:
			File_unlabel_data.write(line)
		data_count += 1
	File_data_Y.close()
	File_data.close()
	File_unlabel_data.close()
	print('Number of training data: ' + str(data_count))
	print('Number of label data in training set: ' + str(labeled_data_count))