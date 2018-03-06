import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import scipy.misc
import numpy as np
import random
import cv2
import multiprocessing

def saveCroppedFace((in_file, out_file, im_size, savegray, idx_file_curr)):
    if idx_file_curr%100==0:
        print idx_file_curr

    classifier_path = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml';

    img = cv2.imread(in_file);
    
    gray  =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade  =  cv2.CascadeClassifier(classifier_path)
    faces  =  face_cascade.detectMultiScale(gray)
    if len(faces)==0:
        print 'PROBLEM';
        return -1
    else:
        sizes=np.array([face_curr[2]*face_curr[3] for face_curr in faces]);
        faces=faces[np.argmax(sizes)];
        size_crop = np.max(sizes)

    [x,y,w,h] = faces;
    
    roi = gray[y:y+h, x:x+w]    
    if not savegray:
        roi = img[y:y+h, x:x+w]

    if im_size is not None:
        roi=cv2.resize(roi,tuple(im_size));
    cv2.imwrite(out_file,roi)

    return size_crop

def save_resized_images((in_file,out_file,im_size,savegray,idx_file_curr)):
    if idx_file_curr%100==0:
        print idx_file_curr

    img = cv2.imread(in_file);
    
    if savegray:
        gray  =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi=cv2.resize(gray,tuple(im_size));
    else:
        roi=cv2.resize(img,tuple(im_size));

    cv2.imwrite(out_file,roi)

    

def script_save_cropped_faces():
    dir_meta= '../data/bp4d'
    out_dir_meta = os.path.join(dir_meta,'preprocess_im_96')
    in_dir_meta = os.path.join(dir_meta,'BP4D','BP4D-training')
    im_list_in = glob.glob(os.path.join(in_dir_meta,'*','*','*.jpg'))

    im_size = [96,96]
    savegray = True
    args = []
    for idx_im_in,im_in in enumerate(im_list_in):
        out_file = im_in.replace(in_dir_meta,out_dir_meta)
        out_dir_curr = os.path.split(out_file)[0]
        # print out_dir_curr
        util.makedirs(out_dir_curr)
        args.append((im_in,out_file,im_size,savegray,idx_im_in))

    print len(args)
    # args = args[:10]
    # for arg in args:
    #     print arg
    #     size = saveCroppedFace(arg)
    #     raw_input()


    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    crop_sizes = pool.map(saveCroppedFace,args)
    content = []
    out_im_all = [arg_curr[1] for arg_curr in args]
    np.savez(os.path.join(dir_meta,'sizes.npz'),crop_sizes = np.array(crop_sizes),out_im_all = np.array(out_im_all))


def make_au_vec_per_frame(csv_file):
    lines = util.readLinesFromFile(csv_file)
    arr = []
    for line in lines:
        arr_curr = [int(val) for val in line.split(',')]
        arr.append(arr_curr)
    
    return np.array(arr)

    


def script_save_resize_faces():
    dir_meta= '../data/bp4d'
    im_size = [110,110]
    out_dir_meta = os.path.join(dir_meta,'preprocess_im_'+str(im_size[0])+'_color_nodetect')
    in_dir_meta = os.path.join(dir_meta,'BP4D','BP4D-training')
    # in_dir_meta = os.path.join(dir_meta,'preprocess_im_'+str(256)+'_color_nodetect')

    im_list_in = glob.glob(os.path.join(in_dir_meta,'*','*','*.jpg'))

    
    savegray = False
    args = []
    for idx_im_in,im_in in enumerate(im_list_in):
        out_file = im_in.replace(in_dir_meta,out_dir_meta)
        if os.path.exists(out_file):
            continue

        out_dir_curr = os.path.split(out_file)[0]
        # print out_dir_curr
        util.makedirs(out_dir_curr)
        args.append((im_in,out_file,im_size,savegray,idx_im_in))

    print len(args)
    # args = args[:10]
    # for arg in args:
        # print arg
        # save_resized_images(arg)
    #     size = saveCroppedFace(arg)
        # raw_input()



    pool = multiprocessing.Pool(4)
    pool.map(save_resized_images,args)
    # crop_sizes = 
    # content = []
    # out_im_all = [arg_curr[1] for arg_curr in args]
    # np.savez(os.path.join(dir_meta,'sizes.npz'),crop_sizes = np.array(crop_sizes),out_im_all = np.array(out_im_all))

def make_anno_files():
    au_keep = [1,2,4,6,7,10,12,14,15,17,23,24]
    out_dir = '../data/bp4d/anno_text'
    util.mkdir(out_dir)
    im_dir_meta = '../data/bp4d/BP4D/BP4D-training'

    # csv_file = '../data/bp4d/AUCoding/AUCoding/F001_T1.csv'
    csv_files = glob.glob('../data/bp4d/AUCoding/AUCoding/*.csv')
    print len(csv_files)
    total_lines = 0
    for idx_csv_file,csv_file in enumerate(csv_files):
        print idx_csv_file
        arr = make_au_vec_per_frame(csv_file)
        # print arr.shape
        # print arr[0,:]
        idx_keep = np.array([1 if val in au_keep else 0 for val in arr[0,:] ])
        idx_keep[0]=1

        # print idx_keep
        # print arr[0,idx_keep>0]


        rel_cols = arr[1:,idx_keep>0]

        # print rel_cols.shape
        
        ims = rel_cols[:,0]
        rest = rel_cols[:,1:]
        assert np.all(np.unique(rest)==np.array([0,1]))

        out_file = os.path.split(csv_file)[1][:-4]
        # print out_file

        subj,sess = out_file.split('_')

        im_dir_curr = os.path.join(im_dir_meta,subj,sess)
        examples = glob.glob(os.path.join(im_dir_curr,'*.jpg'))
        example = os.path.split(examples[0])[1][:-4]
        num_zeros = len(example)
        # print len(example)

        out_lines = []
        for im_idx, im in enumerate(ims):

            im_str = str(im)
            im_str = '0'*(num_zeros-len(im_str))+im_str
            im_file = os.path.join(im_dir_curr,im_str+'.jpg')

            anno = rest[im_idx]

            if not os.path.exists(im_file) or np.sum(anno)<1:
                continue
            
            anno_str = [str(val) for val in anno]
            assert len(anno_str)==len(au_keep)
            out_line = [im_file]+anno_str
            out_line = ' '.join(out_line)
            out_lines.append(out_line)

        total_lines +=len(out_lines)

        out_file_anno = os.path.join(out_dir,out_file+'.txt')
        util.writeFile(out_file_anno,out_lines)
        print out_file_anno,len(out_lines),total_lines


def make_train_test_subs():
    dir_meta = '../data/bp4d'
    im_dir_meta = os.path.join(dir_meta,'BP4D','BP4D-training')
    out_dir_subs = os.path.join(dir_meta,'subs')
    util.mkdir(out_dir_subs)

    subs = [os.path.split(dir_curr)[1] for dir_curr in glob.glob(os.path.join(im_dir_meta,'*'))]
    print subs
    print len(subs)
    subs.sort()
    print subs
    num_splits = 3
    folds = []
    for fold_num in range(num_splits):
        fold_curr = subs[fold_num::num_splits]
        folds.append(fold_curr)
    
    for fold_num in range(num_splits):
        train_folds = []
        for idx_fold,fold_curr in enumerate(folds):
            if idx_fold!=fold_num:
                train_folds = train_folds+fold_curr
        test_folds = folds[fold_num]
        out_file_train = os.path.join(out_dir_subs,'train_'+str(fold_num)+'.txt')
        out_file_test = os.path.join(out_dir_subs,'test_'+str(fold_num)+'.txt')
        assert len(train_folds)+len(test_folds)==len(list(set(train_folds+test_folds)))

        print fold_num, len(train_folds),len(test_folds)
        print out_file_train, out_file_test
        util.writeFile(out_file_train, train_folds)
        util.writeFile(out_file_test, test_folds)


def write_train_file(out_file_train, out_dir_annos, out_dir_im, train_folds, replace_str):
    all_anno_files = []
    for sub_curr in train_folds:
        all_anno_files= all_anno_files+glob.glob(os.path.join(out_dir_annos,sub_curr+'*.txt'))
    
    all_lines = []
    for anno_file in all_anno_files:
        all_lines = all_lines+util.readLinesFromFile(anno_file)

    out_lines = []
    for line_curr in all_lines:
        out_line = line_curr.replace(replace_str,out_dir_im)
        im_out = out_line.split(' ')[0]
        assert os.path.exists(im_out)
        
        out_lines.append(out_line)

    print len(out_lines)
    print out_lines[0]
    random.shuffle(out_lines)
    util.writeFile(out_file_train,out_lines)
    




def make_train_test_files():
    dir_meta = '../data/bp4d'
    out_dir_subs = os.path.join(dir_meta,'subs')
    out_dir_annos = os.path.join(dir_meta, 'anno_text')

    out_dir_im = os.path.join(dir_meta, 'preprocess_im_110_color_nodetect')
    out_dir_files = os.path.join(dir_meta, 'train_test_files_110_color_nodetect')
    replace_str = '../data/bp4d/BP4D/BP4D-training'
    util.mkdir(out_dir_files)
    num_folds = 3

    for fold_num in range(num_folds):
        for file_pre_str in ['train','test']:
            train_sub_file = os.path.join(out_dir_subs,file_pre_str+'_'+str(fold_num)+'.txt')
            train_folds = util.readLinesFromFile(train_sub_file)
            out_file_train = os.path.join(out_dir_files,file_pre_str+'_'+str(fold_num)+'.txt')
            write_train_file(out_file_train, out_dir_annos, out_dir_im, train_folds, replace_str)






def main():
    make_train_test_files()
    # make_train_test_subs()
    # script_save_resize_faces()

    # return
    



        
    



if __name__=='__main__':
    main()