#coding=utf-8
__author__ = "Hai Wang"
import sys,os,time
sys.path.append('code/Data_processing')
from Data_Preprocessing_v import * 
from Feature_selection import *
import Variable_file
from Feature_merge_tree import *
from Feature_merge_tree_NFS import *
from Feature_merge_linear import *
from Feature_merge_linear_NFS import *

def Preprocessing_Finsh():
	#Data preprocessing
	start_time = time.time()

	threshold_path = Variable_file.threshold_path
	Location_path = Variable_file.Location_path
	Source_path = Variable_file.Source_path
	Tree_path = Variable_file.Tree_path
	xgb_group_path = Variable_file.group_path

	print('Data Preprocessing...')
	Preprocessing_v(threshold_path)
	print('Data Preprocessing is complete ')
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
	
	#Feature merge
	print('Feature merge...')
	Feature_merge_linear_NFS()
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2)) 

	#Feature merge
	print('Feature merge...')
	Feature_merge_tree_NFS()
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2)) 

	
	print('(tree)Feature Selection...')
	Feature_selection('_tree_v',Tree_path)
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	print('(tree)Feature Selection...')
	Feature_selection('_group',xgb_group_path)
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	#Feature merge
	print('(tree all)Feature merge...')
	Feature_merge_tree()
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2)) 

	# Location Feature Selection
	print('(Location Feature)Feature Selection...')
	Feature_selection('_Location',Location_path)
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	# Feature Selection
	print('(Source Feature)Feature Selection...')
	Feature_selection('_Source',Source_path)
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

	#Feature merge
	print('Feature merge...')
	Feature_merge_linear()
	print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2)) 

if __name__ == '__main__':

	start_time = time.time()
	Preprocessing_Finsh()
	


	
