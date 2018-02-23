import sys
sys.path.append('./')
from helpers import util, visualize
import os
import numpy as np
import scipy.misc
import sklearn.metrics

def print_accuracy(dir_exp_meta,pre_split,post_split,num_splits):

    all_vals = []
    max_vals = []
    max_val_idx_all = []

    for split_num in range(num_splits):
        dir_curr = os.path.join(dir_exp_meta,pre_split+str(split_num)+post_split)
        log_file = os.path.join(dir_curr,'log.txt')
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

    all_vals = np.array(all_vals)
    print 'RESULTS'
    print dir_exp_meta
    print post_split
    print num_splits
    print ''
    print 'best_idx','best_accu','end_accu'
    for i in range(len(max_val_idx_all)):
        print max_val_idx_all[i],max_vals[i],all_vals[i]
    print ''
    print 'mean', np.mean(all_vals)
    print 'std', np.std(all_vals)
    print 'min', np.min(all_vals)
    print 'max', np.max(all_vals)

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


def get_per_label_accuracy(dir_exp_meta,pre_split,post_split,num_splits,model_num,class_labels):
    predictions_all = []
    labels_all = []
    
    for split_num in range(num_splits):
        dir_curr = os.path.join(dir_exp_meta,pre_split+str(split_num)+post_split)
        out_dir_results = os.path.join(dir_curr,'results_model_'+str(model_num))
        predictions = np.load(os.path.join(out_dir_results, 'predictions.npy'))
        labels = np.load(os.path.join(out_dir_results, 'labels_all.npy'))
        predictions_all.append(predictions)
        labels_all.append(labels)

    predictions_all = np.concatenate(predictions_all,0)
    print predictions_all.shape

    labels_all = np.concatenate(labels_all,0)
    print labels_all.shape

    cm = sklearn.metrics.confusion_matrix(labels_all, predictions_all)
    print cm
    # cm = cm.astype('float') / cm.sum(axis=1)

    print_cm(cm, class_labels)
    out_file = os.path.join(dir_exp_meta,pre_split+post_split[1:]+'_conf_mat.jpg')
    visualize.plot_confusion_matrix(cm, class_labels, out_file,
                          normalize=True)

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


    num_classes = 7
    class_labels = ['Neutral','Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    get_per_label_accuracy(dir_exp_meta,pre_split,post_split,num_splits,model_num,class_labels)

    # print_accuracy(dir_exp_meta,pre_split,post_split,num_splits)





if __name__=='__main__':
    main()