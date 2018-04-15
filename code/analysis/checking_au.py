import sys
sys.path.append('./')
from helpers import util, visualize
import os
import numpy as np
import scipy.misc
import sklearn.metrics
import glob
import multiprocessing
import visualizing_recon_new

dir_server = '/disk3'
str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
click_str = 'http://vision3.idav.ucdavis.edu:1000'

def collate_files(test_dirs):
	label_pre = 'labels_all_'
	pred_pre = 'predictions_'
	labels_all = []
	pred_all = []

	

	for test_dir in test_dirs:
		num_files = glob.glob(os.path.join(test_dir,label_pre+'*.npy'))
		num_files = [int(file_curr[file_curr.rindex('_')+1:file_curr.rindex('.')]) for file_curr in num_files]
		num_files.sort()
		
		for num_curr in num_files:
			labels_all.append(np.load(os.path.join(test_dir, label_pre+str(num_curr)+'.npy')))
			pred_all.append(np.load(os.path.join(test_dir, pred_pre+str(num_curr)+'.npy')))

	labels_all = np.concatenate(labels_all,0)
	pred_all = np.concatenate(pred_all,0)
	return labels_all, pred_all

def compile_and_print_stats(test_dirs,out_file,eer = True):
	labels_all, pred_all = collate_files(test_dirs)
	str_print = []
	pred_bin = pred_all
	
	print 'labels_all.shape', labels_all.shape
	print 'pred_all.shape', pred_all.shape
	# raw_input()
	# pred_bin[pred_bin<=0.5]=0
	# pred_bin[pred_bin>0.5]=1
	# f1_per_class = sklearn.metrics.f1_score(labels_all,pred_bin,average = None)
	# f1_avg= np.mean(f1_per_class)

	if eer:
		f1_per_class, f1_avg, eer_idx, threshold_eer = calculate_f1_curve_eer(labels_all, pred_all)
		print 'eer_idx: %d, threshold_eer: %.2f' % (eer_idx, threshold_eer) 
		str_print.append('eer_idx: %d, threshold_eer: %.2f' % (eer_idx, threshold_eer))
	else:
		pred_bin[pred_bin<=0.5]=0
		pred_bin[pred_bin>0.5]=1
		f1_per_class = sklearn.metrics.f1_score(labels_all,pred_bin,average = None)
		f1_avg= np.mean(f1_per_class)



	print 'f1_per_class' 
	str_print.append('f1_per_class')
	for f1_curr in f1_per_class:
		# print f1_curr
		str_print.append('%.3f' % f1_curr)
		print str_print[-1]
	print ''
	str_print.append( '')

	# print 'f1_avg', f1_avg
	str_print.append( 'f1_avg %.3f' % f1_avg)
	print str_print[-1]
	print ''
	str_print.append( '')


	auc_per_class = sklearn.metrics.roc_auc_score(labels_all, pred_all, average=None)
	auc_avg = np.mean(auc_per_class)
	print 'auc_per_class'
	str_print.append( 'auc_per_class')
	for auc_curr in auc_per_class:
		# print auc_curr
		str_print.append('%.3f' %  auc_curr)
		print str_print[-1]
	print ''
	str_print.append( '')
	# print 'auc_avg', auc_avg
	str_print.append( 'auc_avg %.3f' % auc_avg)
	print str_print[-1]
	print ''
	str_print.append( '')
	# util.writeFile(out_file,str_print)

def calculate_f1_curve_eer(labels_all,pred_all):
	# labels_all, pred_all = collate_files(test_dirs)
	thresholds = np.unique(pred_all)
	thresholds = np.arange(thresholds[0],1.,0.01)
	# print thresholds
	# print thresholds.shape
	# print thresholds.shape
	# thresholds = thresholds[::1000]
	# # print thresholds.shape
	# thresholds = np.arange(0.1,1.0,0.1)

	
	args = []
	for idx_threshold_curr, threshold_curr in enumerate(thresholds):
		args.append((threshold_curr,labels_all,pred_all,idx_threshold_curr))
		

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	prec_recall = pool.map(get_prec_recall,args)
	prec_recall = np.array(prec_recall)
	# print prec_recall.shape
	diff_vals = np.abs(prec_recall[:,0]-prec_recall[:,1])
	eer_idx = np.argmin(diff_vals)
	# print eer_idx, thresholds[eer_idx]

	f1_per_class = get_f1((thresholds[eer_idx],labels_all,pred_all,0))
	f1_avg = np.mean(f1_per_class) 
	# print f1_per_class
	# print np.mean(f1_per_class)
	return f1_per_class, f1_avg, eer_idx, thresholds[eer_idx]

def get_f1((threshold_curr,labels_all,pred_bin,idx_curr)):
	pred_bin = np.array(pred_bin)
	pred_bin[pred_bin<=threshold_curr]=0
	pred_bin[pred_bin>threshold_curr]=1
	f1_per_class = sklearn.metrics.f1_score(labels_all,pred_bin,average = None)
	return f1_per_class

def get_prec_recall((threshold_curr, labels_all,pred_bin,idx_curr)):
	if idx_curr%10==0:
		print idx_curr

	pred_bin[pred_bin<=threshold_curr]=0
	pred_bin[pred_bin>threshold_curr]=1
	prec= sklearn.metrics.precision_score(labels_all, pred_bin, labels=None, pos_label=1, average=None)
	recall = sklearn.metrics.recall_score(labels_all, pred_bin, labels=None, pos_label=1, average=None)

	return np.mean(prec), np.mean(recall)


def script_print_f1_etc():
	# dir_meta = '../experiments/khorrami_capsule_7_3_color3'
	# dir_exp_pre = 'bp4d_110_'
	# dir_exp_post = '_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_50_step_50_0.1_0.001_0.001_0.001_lossweights_1.0_1.0'
	# models_test = [5]


	# dir_meta = '../experiments/khorrami_capsule_7_3_gray3'
	# dir_exp_pre = 'disfa_train_test_8_au_all_method_110_gray_align_'
	# dir_exp_post = '_reconstruct_True_True_flipCrop_marginmulti_False_wdecay_0_20_exp_0.96_350_1e-06_0.001_0.001_0.001_lossweights_1.0_1.0'
	# models_test = [9]

	
	# dir_exp_pre = 'bp4d_train_test_files_110_gray_align_'
	# dir_exp_post = '_reconstruct_True_True_flipCrop_marginmulti_False_wdecay_0_20_exp_0.96_350_1e-06_0.001_0.001_0.001_lossweights_1.0_1.0'
	# models_test = [3]
	# # ,9,14,19]	
	# eer = False
	# type_metric = 'samples'
	# folds = [0,1,2]


	# dir_exp_pre = 'bp4d_train_test_files_110_gray_align_'
	# dir_exp_post = '_reconstruct_True_True_cropkhAugNoColor_marginmulti_False_wdecay_0_10_exp_0.96_350_1e-06_0.001_0.001_0.001_lossweights_1.0_1.0_None'
	# models_test = [9]
	# eer = False
	# folds = [0,1,2]

	
	# dir_exp_pre = 'disfa_train_test_8_au_all_method_110_gray_align_'
	# dir_exp_post = '_reconstruct_True_True_flipCrop_marginmulti_False_wdecay_0_5_exp_0.96_350_1e-06_0_0.001_0.001_lossweights_1.0_1.0_fold_0_epoch_2_fix_exp_correct_mean'
	# models_test = [0]
	# eer = False
	# folds = [0,1,2]

	# dir_meta = '../experiments/vgg_capsule_7_33'
	# dir_exp_pre = 'bp4d_256_train_test_files_256_color_align_'
	# dir_exp_post = '_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_5_exp_0.96_350_1e-06_0_0.001_0.001_lossweights_1.0_1.0'
	# models_test = [0]
	# eer = False
	# type_metric = 'samples'
	# folds = [0,1,2]

	# dir_exp_pre = 'disfa_256_train_test_8_au_all_method_256_color_align_'
	# dir_exp_post = '_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_2_exp_0.96_350_1e-06_0_0.001_0.001_lossweights_1.0_1.0'
	# models_test = [0]
	# _results.txt

	
	# dir_exp_pre = 'disfa_train_test_8_au_all_method_110_gray_align_'
	# dir_exp_post = '_reconstruct_True_True_cropkhAugNoColor_marginmulti_False_wdecay_0_10_exp_0.96_350_1e-06_0_0.001_0.001_lossweights_1.0_1.0_fold_0_epoch_9_moreAug_fix_exp_correct_mean'
	# models_test = [9]
	# eer = False
	# folds = [0,1,2]


	dir_meta = '../experiments/vgg_capsule_7_3_smallrecon3'
	dir_exp_pre = 'bp4d_256_train_test_files_256_color_align_'
	dir_exp_post = '_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_2_exp_0.96_350_1e-06_0_0.001_0.001_lossweights_1.0_0.1'
	models_test = [0]
	folds = [0,1,2]
	eer = False

	dir_exp_pre = 'bp4d_256_train_test_files_256_color_align_'
	dir_exp_post = '_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_5_exp_0.96_350_1e-06_0_0.001_0.001_lossweights_1.0_0.1_True'
	models_test = [0]
	folds = [0,1,2]
	eer = False

	dir_exp_pre = 'bp4d_256_train_test_files_256_color_align_'
	dir_exp_post = '_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_5_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
	models_test = [0]
	folds = [0,1,2]
	eer = False


	dir_meta = '../experiments/vgg_capsule_7_33'
	# dir_exp_pre = 'bp4d_256_train_test_files_256_color_align_'
	# dir_exp_post = '_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
	# models_test = [0]
	# folds = [0,1,2]
	# eer = False

	dir_exp_pre = 'disfa_256_train_test_8_au_all_method_256_color_align_'
	dir_exp_post = '_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
	models_test = [0]
	folds = [0,1,2]
	eer = False

	# dir_meta = '../experiments/khorrami_capsule_7_3_gray3'
	# dir_exp_pre = 'disfa_train_test_8_au_all_method_110_gray_align_'
	# dir_exp_post = '_reconstruct_True_True_cropkhAugNoColor_marginmulti_False_wdecay_0_10_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_1.0_fold_0_epoch_9_moreAug_fix_exp_correct_mean'
	# models_test = [9]
	# folds = [0,1,2]
	# eer = False

	for model_test in models_test:
		out_file = os.path.join(dir_meta,dir_exp_pre+dir_exp_post[1:]+'_model_num_'+str(model_test)+'_'+str(eer)+'.txt')

		test_dirs = []
		for fold_num in folds:
			test_dir = os.path.join(dir_meta,dir_exp_pre+str(fold_num)+dir_exp_post,'results_model_'+str(model_test))
			test_dirs.append(test_dir)
			print test_dir
		print 'fold_num,model_test', folds,model_test
		compile_and_print_stats(test_dirs,out_file, eer)
		# calculate_f1_curve_eer(test_dirs,out_file)
		print '___'


def get_ideal_train_test_file():
	dir_meta = '../experiments/khorrami_capsule_7_3_gray3'
	dir_exp_pre = 'bp4d_train_test_files_110_gray_align_'
	dir_exp_post = '_reconstruct_True_True_cropkhAugNoColor_marginmulti_False_wdecay_0_10_exp_0.96_350_1e-06_0.001_0.001_0.001_lossweights_1.0_1.0_None'
	models_test = [9]
	eer = False
	folds = [0]
	# test_dir = 
	data_dir_meta = '../data/bp4d'
	in_test_file = os.path.join(data_dir_meta, 'train_test_files_110_gray_align','test_0.txt')
	assert os.path.exists(in_test_file)

	out_test_file = os.path.join(data_dir_meta, 'train_test_files_110_gray_align','test_0_best_results.txt')
	test_lines = util.readLinesFromFile(in_test_file)
	print os.path.exists(out_test_file)

	for model_test in models_test:
		out_file = os.path.join(dir_meta,dir_exp_pre+dir_exp_post[1:]+'_model_num_'+str(model_test)+'_'+str(eer)+'.txt')

		test_dirs = []
		for fold_num in folds:
			test_dir = os.path.join(dir_meta,dir_exp_pre+str(fold_num)+dir_exp_post,'results_model_'+str(model_test))
			test_dirs.append(test_dir)
			print test_dir
		print 'fold_num,model_test', folds,model_test
		labels_all, pred_bin = collate_files(test_dirs)
		pred_bin[pred_bin<=0.5]=0
		pred_bin[pred_bin>0.5]=1
		print labels_all.shape,pred_bin.shape
		f1_per_class = sklearn.metrics.f1_score(labels_all,pred_bin,average = None)
		f1_avg= np.mean(f1_per_class)

		bin_true = labels_all==pred_bin
		print bin_true.shape
		bin_true_row = np.sum(bin_true,1)
		print bin_true_row.shape,np.min(bin_true_row),np.max(bin_true_row)
		rows_to_keep = bin_true_row==12
		print rows_to_keep.shape,np.sum(rows_to_keep)
		assert rows_to_keep.size==len(test_lines)
		lines_to_keep = [line_curr for idx_line_curr, line_curr in enumerate(test_lines) if rows_to_keep[idx_line_curr]]
		print len(lines_to_keep)
		util.writeFile(out_test_file,lines_to_keep)








		print f1_avg
		
		# compile_and_print_stats(test_dirs,out_file, eer)
		# calculate_f1_curve_eer(test_dirs,out_file)
		print '___'


def get_auc(pred,gt):

    # print pred.shape
    # print gt.shape
    # print pred
    # print gt
    # ap = []
    # gt[gt>0]=1
    # gt[gt<0]=0
    # print pred
    # print gt

    pred[pred>0.5]=1
    pred[pred<=0.5]=0
    # print pred

    ap = sklearn.metrics.f1_score(gt, pred,average='macro')

    # for idx in range(gt.shape[1]):
    #     ap = ap+[sklearn.metrics.average_precision_score(gt[:,idx], pred[:,idx])]
    return ap

def main():

	out_dir_train = '../experiments/vgg_capsule_7_3_with_dropout3/bp4d_256_train_test_files_256_color_align_2_reconstruct_False_True_all_aug_marginmulti_False_wdecay_0_6_step_6_0.1_0.0001_0.001_0.001_True_0'
	log_file = os.path.join(out_dir_train,'log.txt')
	print log_file
	print util.readLinesFromFile(log_file)
	to_print = []
	for model_test in range(6):
		test_dir = os.path.join(out_dir_train,'results_model_'+str(model_test))
		labels_all, predictions = collate_files([test_dir])
		accuracy = get_auc(predictions,labels_all)
		str_print = 'val accuracy: %.4f' %(accuracy)
		print str_print
		to_print.append(str_print)

	util.writeFile(log_file,to_print)

		

	# get_ideal_train_test_file()

	# return
	# script_print_f1_etc()
	return

	dir_data_meta = '../data/bp4d'
	dir_meta = '../experiments/khorrami_capsule_7_3_gray3'
	train_test_folder = 'train_test_files_110_gray_align'
	dir_exp_pre = 'bp4d_'+train_test_folder+'_'
	dir_exp_post = '_reconstruct_True_True_flipCrop_marginmulti_False_wdecay_0_20_exp_0.96_350_1e-06_0.001_0.001_0.001_lossweights_1.0_1.0'
	model_test = 2
	fold_num = 0
	mean_file = os.path.join(dir_data_meta,train_test_folder,'train_'+str(fold_num)+'_mean.png')
	std_file = os.path.join(dir_data_meta,train_test_folder,'train_'+str(fold_num)+'_std.png')
	au_arr = [1,2,4,6,7,10,12,14,15,17,23,24]

	test_dir = os.path.join(dir_meta,dir_exp_pre+str(fold_num)+dir_exp_post,'results_model_'+str(model_test))
	resize = 96
	visualizing_recon_new.make_html_recon_active_thresh(test_dir,mean_file,std_file, au_arr, resize, thresh_active= 6)

	# raw_input()

	# labels_all, pred_all = collate_files([test_dir])
	# print labels_all.shape, pred_all.shape
	# threshold = 0.5
	# pred_all[pred_all<=threshold] = 0
	# pred_all[pred_all>threshold] = 1
	# f1_avg = np.mean(sklearn.metrics.f1_score(labels_all,pred_all,average = None))
	# print np.unique(pred_all), np.unique(labels_all),f1_avg

	# num_annos = np.sum(labels_all,1)
	# bin_keep = num_annos>=6
	# print np.sum(bin_keep)




	# test_dirs.append(test_dir)
	# print test_dir
	# print 'fold_num,model_test', folds,model_test
	# compile_and_print_stats(test_dirs,out_file, eer)
	# # calculate_f1_curve_eer(test_dirs,out_file)
	# print '___'


if __name__=='__main__':
	main()