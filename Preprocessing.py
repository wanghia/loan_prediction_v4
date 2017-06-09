#coding=utf-8
__author__ = "Hai Wang"
import sys,os,time
sys.path.append('code/Data_processing')
from Data_Preprocessing_v import * 
from LR_Feature_selection import *
from xgb_Feature_selection import *
from Feature_merge_tree import *
from Feature_merge_tree_NFS import *
from Feature_merge_linear import *
from Feature_merge_linear_NFS import *
import Variable_file


def preprocess():
	#Data preprocessing
	start_time = time.time()

	threshold_path = Variable_file.threshold_path
	Location_path = Variable_file.Location_path
	Source_path = Variable_file.Source_path
	Tree_path = Variable_file.Tree_path
	xgb_group_path = Variable_file.group_path

	print("clear 'model/threshold.txt'")
	file_threshold = open(threshold_path,'w')
	file_threshold.close()

	print("clear 'model/xgbfeature.txt'")
	file_tree = open(Tree_path,'w')
	file_tree.close()

	print("clear 'model/Location_feature.txt'")
	file_Location = open(Location_path,'w')
	file_Location.close()

	print("clear 'model/Source_feature.txt'")
	file_Source = open(Source_path,'w')
	file_Source.close()
	
	print("clear 'model/xgbgroup.txt'")
	file_group = open(xgb_group_path,'w')
	file_group.close()

	print('Data Preprocessing...')
	Preprocessing_v(threshold_path)
	print('Data Preprocessing is complete ')
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
	

	
	#Feature merge
	print('Feature merge...')
	Feature_merge_tree_NFS()
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2)) 

	print('(tree_v)Feature Selection...')
	xgb_Feature_selection('tree_v',Tree_path)
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	print('(tree group)Feature Selection...')
	xgb_Feature_selection('group',xgb_group_path)
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	#Feature merge
	print('(tree all)Feature merge...')
	Feature_merge_tree()
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2)) 

	# Location Feature Selection
	print('(Location Feature)Feature Selection...')
	LR_Feature_selection('_Location',Location_path)
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	#Feature merge
	print('Feature merge...')
	Feature_merge_linear_NFS()
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2)) 


	# Feature Selection
	print('(Source Feature)Feature Selection...')
	LR_Feature_selection('_Source',Source_path)
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
	
	#Feature merge
	print('Feature merge...')
	Feature_merge_linear()
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2)) 



if __name__ == '__main__':

	start_time = time.time()
	preprocess()
	


	
