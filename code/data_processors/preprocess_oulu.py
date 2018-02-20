import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import scipy.misc
import numpy as np
import random
def make_train_test_split():
    pass

def make_fold_files():
    dir_meta = '../data/Oulu_CASIA'
    out_dir_files = os.path.join(dir_meta,'train_test_files')
    util.mkdir(out_dir_files)

    dir_meta_subjects = glob.glob(os.path.join(dir_meta,'PreProcess_Img/NI_Acropped','*'))
    range_subjects = ['P'+'0'*(3-len(str(num)))+str(num) for num in range(1,81)]

    num_folds = 10
    folds = []
    for i in range(num_folds):
        fold_curr = range_subjects[i::num_folds]
        folds.append(fold_curr)

    for i in range(num_folds):
        out_file_train = os.path.join(out_dir_files,'train_subs_'+str(i)+'.txt')
        out_file_test = os.path.join(out_dir_files,'test_subs_'+str(i)+'.txt')
        train_subs =[]
        for idx_fold_curr,fold_curr in enumerate(folds):
            if idx_fold_curr!=i:
                train_subs = train_subs+ fold_curr 
        test_subs = folds[i]
        
        util.writeFile(out_file_train,train_subs)
        util.writeFile(out_file_test,test_subs)

def save_mean_std_vals(dir_files):
    im_resize = [96,96]

    for split_num in range(0,10):
        train_file = os.path.join(dir_files,'train_'+str(split_num)+'.txt')
        out_file = os.path.join(dir_files,'train_'+str(split_num)+'_mean_std_val_0_1.npy')

        lines = util.readLinesFromFile(train_file)
        im_all = []
        for line in lines:
            im = line.split(' ')[0]
            im = scipy.misc.imresize(scipy.misc.imread(im),(im_resize[0],im_resize[1])).astype(np.float32)
            im = im/255.
            im = im[:,:,np.newaxis]
            im_all.append(im)

        # print len(im_all)
        print im_all[0].shape
        im_all = np.concatenate(im_all,2)
        

        print im_all.shape, np.min(im_all),np.max(im_all)
        mean_val = np.mean(im_all)
        std_val = np.std(im_all)
        print mean_val,std_val
        mean_std = np.array([mean_val,std_val])
        print mean_std.shape, mean_std
        np.save(out_file,mean_std)

def write_train_test_files():
    dir_meta = '../data/Oulu_CASIA'
    out_dir_files = os.path.join(dir_meta,'train_test_files')
    out_dir_single_im = os.path.join(out_dir_files,'three_im')
    util.mkdir(out_dir_single_im)
    
    expressions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

    
    dir_meta_subjects = glob.glob(os.path.join(dir_meta,'PreProcess_Img/NI_Acropped','*'))

    dir_ims = glob.glob(os.path.join(dir_meta,'PreProcess_Img/NI_Acropped','*','*','*'))

    num_folds = 10
    num_select = 3
    subs_file_pre = 'train_subs_'
    out_file_pre = 'train_'
    

    for fold_num in range(num_folds):
        train_subs = util.readLinesFromFile(os.path.join(out_dir_files,subs_file_pre+str(fold_num)+'.txt'))
        dir_ims_curr = [dir_curr for dir_curr in dir_ims if dir_curr.split('/')[-2] in train_subs]

        out_file_curr = os.path.join(out_dir_single_im,out_file_pre+str(fold_num)+'.txt')
        lines_all = []

        all_lens =[]
        
        for dir_im in dir_ims_curr:
            ims = glob.glob(os.path.join(dir_im,'*.jpeg'))
            all_lens.append( len(ims))
            ims.sort()
            if len(ims)==0:
                print dir_im
            exp = str(expressions.index(dir_im.split('/')[-1])+1)

            neutral = ims[:num_select]
            exp_im = ims[len(ims)-num_select:]
            lines = [im_curr+' 0' for im_curr in neutral]+[im_curr+' '+exp for im_curr in exp_im]
            lines_all.extend(lines)
        
        print min(all_lens),max(all_lens)
        assert len(lines_all) == len(set(lines_all))
        print out_file_curr,len(lines_all)
        random.shuffle(lines_all)
        util.writeFile(out_file_curr,lines_all)

def main():

    dir_meta = '../data/Oulu_CASIA'
    out_dir_files = os.path.join(dir_meta,'train_test_files')
    out_dir_single_im = os.path.join(out_dir_files,'three_im')
    util.mkdir(out_dir_single_im)
    
    save_mean_std_vals(out_dir_single_im)

    # num_folds = 10
    # num_select = 3
    # subs_file_pre = 'train_subs_'
    # out_file_pre = 'train_'
    
    
if __name__=='__main__':
    main()