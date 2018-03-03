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
        

def main():
    create_256_train_test_files()

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