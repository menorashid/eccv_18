import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import scipy.misc
import cv2
import numpy as np

def saveCroppedFace(in_file,out_file,im_size=None,classifier_path=None,savegray=True):
    if classifier_path==None:
        classifier_path = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml';

    img = cv2.imread(in_file);
    gray  =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade  =  cv2.CascadeClassifier(classifier_path)
    faces  =  face_cascade.detectMultiScale(gray)
    if len(faces)==0:
        print 'PROBLEM';
        return in_file;
    else:
        print len(faces),
        sizes=np.array([face_curr[2]*face_curr[3] for face_curr in faces]);
        faces=faces[np.argmax(sizes)];
        print np.max(sizes);

    [x,y,w,h] = faces;
    roi = gray[y:y+h, x:x+w]
    if not savegray:
        roi = img[y:y+h, x:x+w]
    
    if im_size is not None:
        roi=cv2.resize(roi,tuple(im_size));
    cv2.imwrite(out_file,roi)

def saveCKresizeImages():
    anno_file='../data/ck_original/anno_all.txt';
    
    # dir_meta=os.path.join(dir_server,'expression_project/data/ck_96');
    # out_file_html=os.path.join(dir_meta,'check_face.html');
    # replace=False
    # im_size=[96,96];
    # out_dir_meta_meta='../data/ck_'+str(im_size[0])
    dir_server = '/disk3'
    str_replace = ['..','/disk3/maheen_data/eccv_18']
    dir_meta = '../data/ck_256'.replace(str_replace[0],str_replace[1])
    # dir_meta=os.path.join(dir_server,'expression_project/data/ck_192');
    out_file_html=os.path.join(dir_meta,'check_face.html');
    replace=False
    im_size=[256,256];
    out_dir_meta_meta='../data/ck_'+str(im_size[0])

    out_dir_meta=os.path.join(out_dir_meta_meta,'im');
    old_out_dir_meta='../data/ck_original/cohn-kanade-images';
    out_file_anno=os.path.join(out_dir_meta_meta,'anno_all.txt');

    util.makedirs(out_dir_meta);
    old_anno_data=util.readLinesFromFile(anno_file)
    ims=[line_curr.split(' ')[0] for line_curr in old_anno_data];
    problem_cases=[];
    new_anno_data=[];

    # ims=ims[:10];
    for idx_im_curr,im_curr in enumerate(ims):
        print idx_im_curr,
        out_file_curr=im_curr.replace(old_out_dir_meta,out_dir_meta);
        problem=None;
        if not os.path.exists(out_file_curr) or replace:
            out_dir_curr=os.path.split(out_file_curr)[0];
            util.makedirs(out_dir_curr);
            problem=saveCroppedFace(im_curr,out_file_curr,im_size);

        if problem is not None:
            problem_cases.append(problem);
        else:
            new_anno_data.append(old_anno_data[idx_im_curr].replace(old_out_dir_meta,out_dir_meta));

    print len(problem_cases);
    # new_anno_data=[line_curr.replace(old_out_dir_meta,out_dir_meta) for line_curr in old_anno_data];
    util.writeFile(out_file_anno,new_anno_data);

    ims=np.array([line_curr.split(' ')[0].replace(out_dir_meta_meta,dir_meta) for line_curr in new_anno_data]);
    print ims[0];
    im_dirs=np.array([os.path.split(im_curr)[0] for im_curr in ims]);
    im_files=[];
    captions=[];
    for im_dir in np.unique(im_dirs):
        im_files_curr=[util.getRelPath(im_curr, dir_server) for im_curr in ims[im_dirs==im_dir]];
        captions_curr=[os.path.split(im_curr)[1] for im_curr in im_files_curr];
        im_files.append(im_files_curr);
        captions.append(captions_curr);

    visualize.writeHTML(out_file_html,im_files,captions);
    print out_file_html.replace(dir_server,click_str);



def write_facs_file(in_file,out_file,facs_dir):
    im_files= util.readLinesFromFile(in_file)

    out_anno = []

    for im_file in im_files:
        im_file = im_file.split(' ')[0]
        im_name_split = os.path.split(im_file)[1][:-4].split('_')
        facs_file = os.path.join(facs_dir,im_name_split[0],im_name_split[1],'_'.join(im_name_split)+'_facs.txt')

        if not os.path.exists(facs_file):
            continue

        facs_anno = util.readLinesFromFile(facs_file)
        anno = []
        for line_curr in facs_anno:
            line_curr = line_curr.strip().split()
            anno = anno+[str(int(float(val))) for val in line_curr]
        anno_curr = ' '.join([im_file]+anno)
        out_anno.append(anno_curr)

        
    util.writeFile(out_file,out_anno)

def save_mean_std_vals():
    for split_num in range(0,1):
        train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
        test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
        mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
        std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'
        out_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean_std_val_0_1.npy'

        lines = util.readLinesFromFile(train_file)
        im_all = []
        for line in lines:
            im = line.split(' ')[0]
            im = scipy.misc.imread(im).astype(np.float32)
            im = im/255.
            im = im[:,:,np.newaxis]
            im_all.append(im)

        print len(im_all)
        im_all = np.concatenate(im_all,2)
        print im_all.shape, np.min(im_all),np.max(im_all)
        mean_val = np.mean(im_all)
        std_val = np.std(im_all)
        print mean_val,std_val
        mean_std = np.array([mean_val,std_val])
        print mean_std.shape, mean_std
        np.save(out_file,mean_std)

def save_dummy_std_vals():
    for split_num in range(0,10):
        std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std_just_one.png'
        vals = np.ones((96,96))
        cv2.imwrite(std_file,vals)


def create_256_train_test_files():
    in_data_dir = '../data/ck_96/train_test_files'
    out_data_dir = '../data/ck_256/train_test_files'
    util.mkdir(out_data_dir)

    num_folds = 10
    for split_num in range(0,num_folds):
        for file_pre in ['train','test']:
            in_file = os.path.join(in_data_dir,file_pre+'_'+str(split_num)+'.txt')
            out_file = os.path.join(out_data_dir,file_pre+'_'+str(split_num)+'.txt') 
            in_lines = util.readLinesFromFile(in_file)
            # print in_lines[0]
            # raw_input()
            out_lines = [line_curr.replace(in_data_dir,out_data_dir) for line_curr in in_lines]
            print out_file
            util.writeFile(out_file,out_lines)
        

def get_list_of_aus():
    dir_files = '../data/ck_96/train_test_files'
    num_folds = 10
    lines = []
    for file_pre_curr in ['train','test']:
        train_file = os.path.join(dir_files,file_pre_curr+'_facs_'+str(0)+'.txt')
        lines = lines+util.readLinesFromFile(train_file)

    aus_all = []
    for line in lines:
        # print line
        aus = [int(val) for val in line.split(' ')[1:]]
        # print aus
        aus = aus[0::2]
        # print aus
        # raw_input()
        aus_all += aus 

    print set(aus_all)

    list_keep = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 24, 25, 26, 27, 38]
    print len(list_keep)
    assert all(list_keep[i] <= list_keep[i+1] for i in xrange(len(list_keep)-1))

    count_percent_left = []
    for val in set(aus_all):
        if val in list_keep:
            count_val = aus_all.count(val)
            count_percent = count_val/float(len(aus_all))
            print val, count_val, len(aus_all), count_percent
        else:
            count_val = aus_all.count(val)
            count_percent = count_val/float(len(aus_all))
            if count_percent>0.01:
                print 'leaving',val,count_percent
            count_percent_left.append(count_percent)

    print np.min(count_percent_left),np.max(count_percent_left)

def merge_emo_facs(emo_file,facs_file,out_file,list_au_keep,idx_map):

    assert os.path.exists(emo_file)
    assert os.path.exists(facs_file)
    
    emo_lines = util.readLinesFromFile(emo_file)
    facs_lines = util.readLinesFromFile(facs_file)

    au_bin = np.zeros((len(emo_lines),np.max(idx_map)+1))
    print 'au_bin.shape', au_bin.shape

    emo_ims = [line.split(' ')[0] for line in emo_lines]
    facs_ims = [line.split(' ')[0] for line in facs_lines]
    for idx_facs,facs_im in enumerate(facs_ims):

        idx_emo = emo_ims.index(facs_im)
        facs = facs_lines[idx_facs]
        
        facs = [int(val) for val in facs.split(' ')[1:]]
        facs = facs[::2]
        found = 0
        for facs_curr in facs:
            if facs_curr in list_au_keep:
                found = 1
                idx_au = list_au_keep.index(facs_curr)
                au_bin[idx_emo,idx_map[idx_au]] = 1

        if not found:
            print facs_lines[idx_facs]
            print emo_lines[idx_emo]
            raw_input()
    
    facs_bin = np.sum(au_bin,axis=1,keepdims=True)
    print facs_bin.shape,np.min(facs_bin),np.max(facs_bin),np.sum(facs_bin>0)
    facs_bin[facs_bin>0] = 1
    
    print np.sum(facs_bin),len(facs_lines),np.sum(facs_bin)==len(facs_lines)
    print np.sum(au_bin,0)
    
    out_mat = np.concatenate((facs_bin,au_bin),1)
    print out_mat.shape
    assert out_mat.shape[0]==len(emo_lines)
    assert out_mat.shape[1]==np.max(idx_map)+2
    
    if os.path.split(out_file)[1].startswith('train'):
        assert np.all(np.sum(au_bin,0)>0)
    

    out_lines = []
    for idx_emo_line,emo_line in enumerate(emo_lines):
        facs_arr_str = [str(int(val)) for val in list(out_mat[idx_emo_line])]
        out_line = emo_line+' '+' '.join(facs_arr_str)
        # print out_line
        # raw_input()
        out_lines.append(out_line)

    print out_file
    util.writeFile(out_file,out_lines)


        


def make_combo_train_test_files():
    dir_files = '../data/ck_96/train_test_files'
    num_folds = 10
    list_au_keep = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 24, 25, 26, 27, 38]
    assert all(list_au_keep[i] < list_au_keep[i+1] for i in xrange(len(list_au_keep)-1))

    idx_26 = list_au_keep.index(26)
    idx_map = range(idx_26)+[idx_26,idx_26]
    print idx_map
    idx_map = idx_map + range(len(idx_map)-1,len(list_au_keep)-1)
    print idx_map
    
    assert len(idx_map)==len(list_au_keep)
    for i in range(len(idx_map)):
        print idx_map[i],list_au_keep[i]

    # raw_input()
    

    for fold_curr in range(num_folds):
        for file_pre in ['train','test']:
            fold_curr = str(fold_curr)
            emo_file = os.path.join(dir_files, '_'.join([file_pre,fold_curr+'.txt']))
            facs_file = os.path.join(dir_files, '_'.join([file_pre,'facs',fold_curr+'.txt']))
            out_file = os.path.join(dir_files, '_'.join([file_pre,'emofacscombo',fold_curr+'.txt']))
            merge_emo_facs(emo_file,facs_file,out_file,list_au_keep,idx_map)



def main():
    # get_list_of_aus()
    # list_au_keep = [1, 2, 4, 5, 6, 7, 9, 12, 14, 15, 16, 20, 23, 26]
    # list_au_keep.sort()

    make_combo_train_test_files()

    # save_dummy_std_vals()
    # create_256_train_test_files()

    # saveCKresizeImages()
    return

    data_dir = '../data/ck_96/train_test_files'
    train_file = os.path.join(data_dir,'train_0.txt')
    
    train_data = util.readLinesFromFile(train_file)
    train_data = [int(line_curr.split(' ')[1]) for line_curr in train_data]
    print set(train_data)

    return
    facs_anno_dir = '../data/ck_original/FACS'



    all_files = []
    fold_num = 0
    for fold_num in range(10):
        for file_pre in ['train','test']:
            in_file = os.path.join(data_dir,file_pre+'_'+str(fold_num)+'.txt')
            out_file = os.path.join(data_dir,file_pre+'_facs_'+str(fold_num)+'.txt')
            write_facs_file(in_file,out_file,facs_anno_dir)
            print in_file,out_file



if __name__=='__main__':
    main()