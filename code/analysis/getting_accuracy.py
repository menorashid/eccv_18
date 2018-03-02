import sys
sys.path.append('./')
from helpers import util, visualize
import os
import numpy as np
import scipy.misc
import sklearn.metrics

def print_accuracy(dir_exp_meta,pre_split,post_split,range_splits,log='log.txt'):

    all_vals = []
    max_vals = []
    max_val_idx_all = []

    for split_num in range_splits:
        dir_curr = os.path.join(dir_exp_meta,pre_split+str(split_num)+post_split)
        log_file = os.path.join(dir_curr,log)
        log_lines = util.readLinesFromFile(log_file)
        log_lines = [line for line in log_lines if 'val accuracy' in line]
        val_accuracy = [float(line.split(' ')[-1]) for line in log_lines]
        val_accuracy = np.array(val_accuracy)
        # max_val_idx = np.argmax(np.array(val_accuracy))
        # val_accuracy = max(np.array(val_accuracy))
        
        # [-1]
        all_vals.append(val_accuracy[-1])
        max_vals.append(np.max(val_accuracy))
        max_val_idx_all.append(np.argmax(val_accuracy))

    str_print = []
    all_vals = np.array(all_vals)
    str_print.append('RESULTS')
    # str_print.append(dir_exp_meta)
    str_print.append(post_split)
    # str_print.append(range_splits)
    # str_print.append('')
    # str_print.append(' '.join([str(val) for val in ['best_idx','best_accu','end_accu']]))
    # for i in range(len(max_val_idx_all)):
    #     str_print.append(' '.join([str(val) for val in [max_val_idx_all[i],max_vals[i],all_vals[i]]]))
    # str_print.append('')
    str_print.append(' '.join([str(val) for val in ['mean', np.mean(all_vals)]]))
    str_print.append(' '.join([str(val) for val in ['std', np.std(all_vals)]]))
    str_print.append(' '.join([str(val) for val in ['min', np.min(all_vals)]]))
    str_print.append(' '.join([str(val) for val in ['max', np.max(all_vals)]]))
    str_print.append(' '.join([str(val) for val in ['mean max', np.mean(max_vals)]]))
    str_print = [str(val) for val in str_print]
    for val in str_print:
        print val
    out_file = os.path.join(dir_exp_meta,pre_split+post_split[1:]+'_results.txt')
    util.writeFile(out_file,str_print)
    print out_file

    

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    cm = cm.astype('float') / cm.sum(axis=1,keepdims=True)
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.3f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print


def get_per_label_accuracy(dir_exp_meta,pre_split,post_split,range_splits,model_num,class_labels):
    predictions_all = []
    labels_all = []
    
    for split_num in range_splits:
        dir_curr = os.path.join(dir_exp_meta,pre_split+str(split_num)+post_split)
        out_dir_results = os.path.join(dir_curr,'results_model_'+str(model_num))
        predictions = np.load(os.path.join(out_dir_results, 'predictions.npy'))
        out_all = np.load(os.path.join(out_dir_results, 'out_all.npy'))
        labels = np.load(os.path.join(out_dir_results, 'labels_all.npy'))
        predictions_all.append(predictions)
        labels_all.append(labels)

    predictions_all = np.concatenate(predictions_all,0)
    # print predictions_all.shape

    labels_all = np.concatenate(labels_all,0)
    # print labels_all.shape

    cm = sklearn.metrics.confusion_matrix(labels_all, predictions_all)
    # print cm
    # cm = cm.astype('float') / cm.sum(axis=1)

    # print_cm(cm, class_labels)
    out_file = os.path.join(dir_exp_meta,pre_split+post_split[1:]+'_conf_mat.jpg')
    visualize.plot_confusion_matrix(cm, class_labels, out_file,
                          normalize=True)
    print out_file.replace('..','http://vision3.idav.ucdavis.edu:1000/maheen_data/eccv_18')


def view_loss_curves(dir_exp_meta,pre_split,post_split,range_splits,model_num):
    dir_server = '/disk3'
    str_replace = ['..','/disk3/maheen_data/eccv_18']
    out_file_html = os.path.join(dir_exp_meta,pre_split+post_split[1:]+'_loss_curves.html').replace(str_replace[0],str_replace[1])

    ims_html = []
    captions_html = []

    for split_num in range_splits:
        caption = [str(split_num)]
        dir_curr = os.path.join(dir_exp_meta,pre_split+str(split_num)+post_split)
        loss_file = os.path.join(dir_curr, 'loss.jpg')
        dirs_res = [os.path.join(dir_curr,'results_model_'+str_curr) for str_curr in [str(model_num),str(model_num)+'_center']]
        dirs_res = [dir_curr for dir_curr in dirs_res if os.path.exists(dir_curr)]
        for dir_res in dirs_res:
            log_file = os.path.join(dir_res,'log.txt')
            val_accuracy = util.readLinesFromFile(log_file)[-1]
            val_accuracy = val_accuracy.split(' ')[-1]
            caption.append(val_accuracy)
        caption = ' '.join(caption)
        im_curr = util.getRelPath(loss_file.replace(str_replace[0],str_replace[1]),dir_server)
        ims_html.append([im_curr])
        captions_html.append([caption])

    visualize.writeHTML(out_file_html,ims_html,captions_html,200,200)
    print out_file_html.replace(dir_server,'http://vision3.idav.ucdavis.edu:1000')

def main():

    dir_exp_meta = '../experiments/khorrami_caps_k7_s3_oulu'
    pre_split = 'oulu_single_'
    post_split ='_all_aug_nopool_300_step_150_0.1_0.001'
    num_splits = 10
    model_num = 299

    dir_exp_meta = '../experiments/caps_heavy_48'
    pre_split = 'oulu_single_'
    post_split = '_all_aug_nopool_200_step_100_0.1_0.001'
    num_splits = 10
    model_num = 199

    # dir_exp_meta = '../experiments/khorrami_caps_k7_s3_oulu_class_weights'
    # pre_split = 'oulu_single_im_'
    # post_split = '_all_aug_nopool_300_step_150_0.1_0.01'
    # num_splits = 1
    # model_num = 299

    # dir_exp_meta = '../experiments/khorrami_caps_k7_s3_oulu_class_weights'
    # pre_split = 'oulu_single_im_'
    # post_split = '_all_aug_nopool_300_step_150_0.1_0.0005'
    # num_splits = 1
    # model_num = 299

    dir_exp_meta = '../experiments/khorrami_caps_k7_s3_oulu_spread_0.2'
    pre_split = 'oulu_single_im_'
    post_split = '_all_aug_nopool_300_step_150_0.1_0.001'
    num_splits = 1
    model_num = 299

    dir_exp_meta = '../experiments/khorrami_caps_k7_s3_oulu_spread_0.2_vl_gray_r_1_init/'
    pre_split = 'oulu_three_im_no_neutral_just_strong_'
    post_split = '_all_aug_max_300_step_300_0.1_0.001'
    num_splits = 10
    model_num = 299

    num_classes = 6
    class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    
    

    ######
    dir_exp_meta = '../experiments/khorrami_caps_k7_s3_oulu_spread_0.2_vl_gray_r_3_init_correct_out'
    num_splits = range(10)

    dir_exp_meta = '../experiments/khorrami_caps_k7_s3_oulu_spread_0.2_vl_gray_r_2_init_correct_out'
    num_splits = [0,1]
    
    dir_exp_meta = '../experiments/khorrami_caps_k7_s3_oulu_spread_0.2_vl_gray_r_1_init_correct_out'
    num_splits = range(10)
    
    pre_split = 'oulu_three_im_no_neutral_just_strong_'
    post_split = '_all_aug_max_300_step_300_0.1_0.001'
    model_num = 299
    ######

    dir_exp_meta = '../experiments/oulu_r3_hopeful'
    num_splits = range(10)


    # dir_exp_meta = '../experiments/khorrami_caps_k7_s3_oulu_spread_0.2_vl_gray_r_2_init_correct_out'
    # num_splits = [0,1]
    
    # dir_exp_meta = '../experiments/khorrami_caps_k7_s3_oulu_spread_0.2_vl_gray_r_1_init_correct_out'
    # num_splits = range(10)
    
    pre_split = 'oulu_three_im_no_neutral_just_strong_True_'
    post_split = '_wdecay_all_aug_max_500_step_500_0.1_0.0001'
    
    # for model_num in ['499','499_center']:
    # ,'bestVal','bestVal_center']:
        # print 'MODEL TYPE',model_num
        # log = 'log_test_center.txt' if model_num.endswith('_center') else 'log_test.txt'
        # get_per_label_accuracy(dir_exp_meta,pre_split,post_split,num_splits,model_num,class_labels)
        # print_accuracy(dir_exp_meta,pre_split,post_split,num_splits,log = log)
        # view_loss_curves(dir_exp_meta,pre_split,post_split,num_splits,model_num)

    ####

    dir_exp_meta = '../experiments/oulu_vgg_r1_noinit_preprocessed'
    num_splits = range(10)


    # dir_exp_meta = '../experiments/khorrami_caps_k7_s3_oulu_spread_0.2_vl_gray_r_2_init_correct_out'
    # num_splits = [0,1]
    
    # dir_exp_meta = '../experiments/khorrami_caps_k7_s3_oulu_spread_0.2_vl_gray_r_1_init_correct_out'
    # num_splits = range(10)
    pre_splits = ['oulu_three_im_no_neutral_just_strong_False_']*3
    model_names = ['vgg_capsule_disfa','vgg_capsule_disfa_bigprimary','vgg_capsule_disfa_bigclass']
    post_splits = ['_'+model_name+'_all_aug_wdecay_0_50_step_50_0.1_1e-05_0.0001' for model_name in model_names]
    
    
    for pre_split,post_split in zip(pre_splits,post_splits):
        for model_num in ['49']:
            print 'MODEL TYPE',model_num
            get_per_label_accuracy(dir_exp_meta,pre_split,post_split,num_splits,model_num,class_labels)
            print_accuracy(dir_exp_meta,pre_split,post_split,num_splits)
            view_loss_curves(dir_exp_meta,pre_split,post_split,num_splits,model_num)





if __name__=='__main__':
    main()