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
    out_dir_meta = os.path.join(dir_meta,'preprocess_im_256_color_nodetect')
    in_dir_meta = os.path.join(dir_meta,'BP4D','BP4D-training')
    im_list_in = glob.glob(os.path.join(in_dir_meta,'*','*','*.jpg'))

    im_size = [256,256]
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
        
def main():
    script_save_resize_faces()
    
    return
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



        
    



if __name__=='__main__':
    main()