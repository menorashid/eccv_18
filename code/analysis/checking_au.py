import sys
sys.path.append('./')
from helpers import util, visualize
import os
import numpy as np
import scipy.misc
import sklearn.metrics
import glob

dir_server = '/disk3'
str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
click_str = 'http://vision3.idav.ucdavis.edu:1000'


def compile_and_print_stats(test_dirs,out_file,type_metric=None):
	label_pre = 'labels_all_'
	pred_pre = 'predictions_'
	labels_all = []
	pred_all = []

	str_print = []

	for test_dir in test_dirs:
		num_files = glob.glob(os.path.join(test_dir,label_pre+'*.npy'))
		num_files = [int(file_curr[file_curr.rindex('_')+1:file_curr.rindex('.')]) for file_curr in num_files]
		num_files.sort()
		
		for num_curr in num_files:
			labels_all.append(np.load(os.path.join(test_dir, label_pre+str(num_curr)+'.npy')))
			pred_all.append(np.load(os.path.join(test_dir, pred_pre+str(num_curr)+'.npy')))

	labels_all = np.concatenate(labels_all,0)
	pred_all = np.concatenate(pred_all,0)
	
	pred_bin = pred_all
	pred_bin[pred_bin<=0.5]=0
	pred_bin[pred_bin>0.5]=1

	print 'labels_all.shape', labels_all.shape
	print 'pred_all.shape', pred_all.shape

	f1_per_class = sklearn.metrics.f1_score(labels_all,pred_bin,average = type_metric)
	print f1_per_class
	raw_input()
	f1_avg= np.mean(f1_per_class)

	print 'f1_per_class' 
	str_print.append('f1_per_class')
	for f1_curr in f1_per_class:
		print f1_curr
		str_print.append('%.3f' % f1_curr)
	print ''
	str_print.append( '')

	print 'f1_avg', f1_avg
	str_print.append( 'f1_avg %.3f' % f1_avg)
	print ''
	str_print.append( '')


	auc_per_class = sklearn.metrics.roc_auc_score(labels_all, pred_all, average=None)
	auc_avg = np.mean(auc_per_class)
	print 'auc_per_class'
	str_print.append( 'auc_per_class')
	for auc_curr in auc_per_class:
		print auc_curr
		str_print.append('%.3f' %  auc_curr)
	print ''
	str_print.append( '')
	print 'auc_avg', auc_avg
	str_print.append( 'auc_avg %.3f' % auc_avg)
	print ''
	str_print.append( '')
	util.writeFile(out_file,str_print)







def main():
	# dir_meta = '../experiments/khorrami_capsule_7_3_color3'
	# dir_exp_pre = 'bp4d_110_'
	# dir_exp_post = '_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_50_step_50_0.1_0.001_0.001_0.001_lossweights_1.0_1.0'
	# models_test = [5]


	dir_meta = '../experiments/khorrami_capsule_7_3_gray3'
	dir_exp_pre = 'disfa_train_test_8_au_all_method_110_gray_align_'
	dir_exp_post = '_reconstruct_True_True_flipCrop_marginmulti_False_wdecay_0_20_exp_0.96_350_1e-06_0.001_0.001_0.001_lossweights_1.0_1.0'
	models_test = [9]

	
	dir_exp_pre = 'bp4d_train_test_files_110_gray_align_'
	dir_exp_post = '_reconstruct_True_True_flipCrop_marginmulti_False_wdecay_0_20_exp_0.96_350_1e-06_0.001_0.001_0.001_lossweights_1.0_1.0'
	models_test = [2]	
	type_metric = 'samples'
	
	folds = [0,1,2]

		
	for model_test in models_test:
		out_file = os.path.join(dir_meta,dir_exp_pre+dir_exp_post[1:]+'_model_num_'+str(models_test)+'_'+type_metric+'.txt')

		test_dirs = []
		for fold_num in folds:
			test_dir = os.path.join(dir_meta,dir_exp_pre+str(fold_num)+dir_exp_post,'results_model_'+str(model_test))
			test_dirs.append(test_dir)
			print test_dir
		print 'fold_num,model_test', folds,model_test
		compile_and_print_stats(test_dirs,out_file,type_metric = type_metric)
		print '___'


	return
	print 'hello'

	dir_curr = '../experiments/khorrami_capsule_7_3_color3/bp4d_256_0_reconstruct_False_True_all_aug_marginmulti_False_wdecay_0_100_step_100_0.1_0.001_0.001_0.001_lossweights_1.0_1.0/results_model_10'
	gt = np.load(os.path.join(dir_curr,'labels_all.npy'))
	pred = np.load(os.path.join(dir_curr,'predictions.npy'))

	pred[pred>0.5]=1
	pred[pred<=0.5]=0

	ap = sklearn.metrics.f1_score(gt, pred,average='weighted')
	print ap
	print np.mean(ap)
	print gt.shape
	print pred.shape
	print gt[:10]
	print pred[:10]

if __name__=='__main__':
	main()