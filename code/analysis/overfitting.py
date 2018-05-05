import sys
sys.path.append('./')
import numpy as np
import scipy.misc
import cv2
import os
import glob
from helpers import util,visualize,augmenters


def overfitting():
	out_dir = '../experiments/figures/overfitting'
	util.makedirs(out_dir)

	# dirs = []
	# dir_meta = '../experiments/showing_overfitting_justhflip_khorrami_capsule_7_31'
	# dir_curr = os.path.join(dir_meta,'oulu_96_three_im_no_neutral_just_strong_False_0_reconstruct_False_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001')
	# dirs.append(dir_curr)
	# dir_meta = '../experiments/showing_overfitting_justhflip_khorrami_capsule_7_33'
	# dir_r3 = 'oulu_96_three_im_no_neutral_just_strong_False_0_reconstruct_False_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001'
	# dir_r3_lw_eq = 'oulu_96_three_im_no_neutral_just_strong_False_0_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001'
	# dir_r3_lw_b = 'oulu_96_three_im_no_neutral_just_strong_False_0_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001_lossweights_1.0_100.0'

	dirs = []
	# dir_meta = '../experiments/showing_overfitting_justhflip_khorrami_capsule_7_31'
	# dir_curr = os.path.join(dir_meta,'oulu_96_three_im_no_neutral_just_strong_False_9_reconstruct_False_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001')
	# dirs.append(dir_curr)
	dir_meta = '../experiments/showing_overfitting_justhflip_khorrami_capsule_7_33'
	dir_r3 = 'oulu_96_three_im_no_neutral_just_strong_False_9_reconstruct_False_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001'
	dir_r3_lw_eq = 'oulu_96_three_im_no_neutral_just_strong_False_9_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001_lossweights_1.0_1.0'
	dir_r3_lw_b = 'oulu_96_three_im_no_neutral_just_strong_False_9_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001_lossweights_1.0_100.0'

	dir_r3_do = '../experiments/showing_overfitting_justhflip_khorrami_capsule_7_3_with_dropout3/oulu_96_three_im_no_neutral_just_strong_False_9_reconstruct_False_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.5'


	dirs_to_pend = [dir_r3,dir_r3_lw_eq,dir_r3_lw_b]






	for dir_curr in dirs_to_pend:
		dir_curr = os.path.join(dir_meta, dir_curr)
		dirs.append(dir_curr)

	dirs.append(dir_r3_do)

	window = 10
	val_lim = 600
	epoch_range = range(window-1,val_lim)
	
	out_file = os.path.join(out_dir,'val_accuracy_9_do.png')
	xAndYs = []
	legend_entries = ['R3+0','R3+1e-7','R3+1e-5','R3+DO']
	for dir_curr in dirs:
		log_file_curr = os.path.join(dir_curr,'log.txt')
		val_losses = [line_curr for line_curr in util.readLinesFromFile(log_file_curr) if 'val accuracy' in line_curr]
		val_losses = [float(line_curr.split(' ')[-1]) for line_curr in val_losses]
		val_losses = val_losses[:val_lim]
		print len(val_losses)
				
		val_losses = np.convolve(val_losses, np.ones((window,))/window, mode='valid')

		xAndYs.append((epoch_range,val_losses))
	visualize.plotSimple(xAndYs,out_file=out_file,xlabel='Epoch',ylabel='Validation Accuracy',legend_entries= legend_entries,ylim=[0.6,0.8],outside=True)


def overfitting_do():
	out_dir = '../experiments_dropout/figures/overfitting'
	util.makedirs(out_dir)

	dir_meta_r1 = '../experiments_dropout/khorrami_capsule_7_3_bigclass_with_dropout_1'
	dir_meta_r3 = '../experiments_dropout/khorrami_capsule_7_3_bigclass_with_dropout_3'
	dirs_post = ['oulu_96_train_test_files_preprocess_vl_9_reconstruct_False_none_600_step_600_0.1_0.001_0.001_0.0',
	'oulu_96_train_test_files_preprocess_vl_9_reconstruct_False_none_600_step_600_0.1_0.001_0.001_0.5',
	'oulu_96_train_test_files_preprocess_vl_9_reconstruct_False_hs_flip_600_step_600_0.1_0.001_0.001_0.0',
	'oulu_96_train_test_files_preprocess_vl_9_reconstruct_False_hs_flip_600_step_600_0.1_0.001_0.001_0.5']
	meta_legend_entries = ['R1','R3']
	sub_legend_entries = ['None 0','None 0.5','HS+Flip 0','HS+Flip 0.5']
	dirs = [os.path.join(dir_meta_curr,dir_curr) for dir_meta_curr in [dir_meta_r1,dir_meta_r3] for dir_curr in dirs_post]
	legend_entries = [meta_legend+' '+sub_legend for meta_legend in meta_legend_entries for sub_legend in sub_legend_entries]


	# # dirs = []
	# # dir_meta = '../experiments/showing_overfitting_justhflip_khorrami_capsule_7_31'
	# # dir_curr = os.path.join(dir_meta,'oulu_96_three_im_no_neutral_just_strong_False_0_reconstruct_False_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001')
	# # dirs.append(dir_curr)
	# # dir_meta = '../experiments/showing_overfitting_justhflip_khorrami_capsule_7_33'
	# # dir_r3 = 'oulu_96_three_im_no_neutral_just_strong_False_0_reconstruct_False_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001'
	# # dir_r3_lw_eq = 'oulu_96_three_im_no_neutral_just_strong_False_0_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001'
	# # dir_r3_lw_b = 'oulu_96_three_im_no_neutral_just_strong_False_0_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001_lossweights_1.0_100.0'

	# dirs = []
	# # dir_meta = '../experiments/showing_overfitting_justhflip_khorrami_capsule_7_31'
	# # dir_curr = os.path.join(dir_meta,'oulu_96_three_im_no_neutral_just_strong_False_9_reconstruct_False_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001')
	# # dirs.append(dir_curr)
	# dir_meta = '../experiments/showing_overfitting_justhflip_khorrami_capsule_7_33'
	# dir_r3 = 'oulu_96_three_im_no_neutral_just_strong_False_9_reconstruct_False_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001'
	# dir_r3_lw_eq = 'oulu_96_three_im_no_neutral_just_strong_False_9_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001_lossweights_1.0_1.0'
	# dir_r3_lw_b = 'oulu_96_three_im_no_neutral_just_strong_False_9_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001_lossweights_1.0_100.0'

	# dir_r3_do = '../experiments/showing_overfitting_justhflip_khorrami_capsule_7_3_with_dropout3/oulu_96_three_im_no_neutral_just_strong_False_9_reconstruct_False_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.5'


	# dirs_to_pend = [dir_r3,dir_r3_lw_eq,dir_r3_lw_b]






	# for dir_curr in dirs_to_pend:
	# 	dir_curr = os.path.join(dir_meta, dir_curr)
	# 	dirs.append(dir_curr)

	# dirs.append(dir_r3_do)

	window = 10
	val_lim = 600
	epoch_range = range(window-1,val_lim)
	
	dirs = dirs[:4]+dirs[-2:]

	out_file = os.path.join(out_dir,'val_accuracy_9_do.png')
	xAndYs = []
	# legend_entries = ['R3+0','R3+1e-7','R3+1e-5','R3+DO']
	for dir_curr in dirs:
		log_file_curr = os.path.join(dir_curr,'log.txt')
		val_losses = [line_curr for line_curr in util.readLinesFromFile(log_file_curr) if 'val accuracy' in line_curr]
		val_losses = [float(line_curr.split(' ')[-1]) for line_curr in val_losses]
		val_losses = val_losses[:val_lim]
		print dir_curr, len(val_losses)
				
		val_losses = np.convolve(val_losses, np.ones((window,))/window, mode='valid')

		xAndYs.append((epoch_range,val_losses))
	visualize.plotSimple(xAndYs,out_file=out_file,xlabel='Epoch',ylabel='Validation Accuracy',legend_entries= legend_entries,ylim=[0.6,0.8],outside=True)


def collate_labels(dir_curr,num_it = False):
	if num_it:
		print dir_curr
		labels_all = glob.glob(os.path.join(dir_curr,'labels_all_*.npy'))
		labels_all.sort()
		num_labels = len(labels_all)
		preds_all = glob.glob(os.path.join(dir_curr,'predictions_*.npy'))
		preds_all.sort()
		assert len(preds_all)==len(labels_all)
		labels = [np.load(file_curr) for file_curr in labels_all]
		preds = [np.load(file_curr) for file_curr in preds_all]
		labels = np.concatenate(labels,0)
		preds = np.concatenate(preds,0)
	else:
		labels = np.load(os.path.join(dir_curr,'labels_all.npy'))
		preds = np.load(os.path.join(dir_curr,'predictions.npy'))
				
	# print labels.shape, preds.shape
	accu =  np.sum(labels==preds)/float(labels.size)
	return labels, preds, accu

def main():
	overfitting_do()
	# overfitting()
	return

	# ck_stuff
	meta_us = '../experiments/khorrami_capsule_7_3_bigclass3'
	us_pre = 'ck_96_train_test_files_non_peak_one_third_'
	us_post = '_reconstruct_True_True_all_aug_margin_False_wdecay_0_300_exp_0.96_350_1e-06_0.001_0.001_0.001'
	out_dir = '../experiments/figures/ck_intensity_exp'
	util.mkdir(out_dir)

	meta_them = '../experiments/khorrami_ck_96_caps_bl'
	them_pre = 'ck_'
	them_post = '_train_test_files_non_peak_one_third_khorrami_ck_96_300_exp_0.96_350_1e-06_0.001_0.001'


	folds = range(10)
	model_range = range(0,300,50)+[299]
	post_res_range = ['','_easy']

	legend_entries = ['Ours Hard','Ours Easy','BL Hard','BL Easy']

	# for fold_curr in folds:
	# 	accus = [[] for i in range(4)]
	
	# 	for idx_post_res, post_res in enumerate(post_res_range):
	# 		for model_curr in model_range:
	# 			dir_res_us = os.path.join(meta_us, us_pre+str(fold_curr)+us_post,'results_model_'+str(model_curr)+post_res)
	# 			_,_, accu_us = collate_labels(dir_res_us,num_it=True)

	# 			dir_res_them = os.path.join(meta_them, them_pre+str(fold_curr)+them_post,'results_model_'+str(model_curr)+post_res)
	# 			_,_, accu_them = collate_labels(dir_res_them,num_it=False)

	# 			accus[idx_post_res].append(accu_us)
	# 			accus[idx_post_res+2].append(accu_them)

	# 	out_file_curr = os.path.join(out_dir,'fold_'+str(fold_curr)+'.png')
	# 	xAndYs = [(model_range,arr_curr) for arr_curr in accus]
	# 	print out_file_curr
	# 	visualize.plotSimple(xAndYs, out_file = out_file_curr,ylabel = 'Accuracy',xlabel='Epoch',legend_entries = legend_entries, title='Fold '+str(fold_curr),outside=True)

	visualize.writeHTMLForFolder(out_dir,'.png')



				# raw_input()






if __name__=='__main__':
	main()