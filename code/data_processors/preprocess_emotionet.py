# import cv2
import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import scipy.misc
import numpy as np
import random
import face_alignment
import skimage.transform
import multiprocessing
import dlib
import cv2
import shutil
import urllib


def download_image((url,out_file,idx)):
    if idx%100==0:
        print idx
    try:
        urllib.urlretrieve(url, out_file)
    except:
        print 'ERROR',url,out_file


def script_download_image():
    # idx_url_file,url_file,url_files,str_replace):
    dir_meta = '../data/emotionet'
    out_dir_im = os.path.join(dir_meta,'im')
    util.mkdir(out_dir_im)
    str_replace = ['http://cbcsnas01.ece.ohio-state.edu/EmotioNet/Images',out_dir_im]

    dir_url_files = os.path.join(dir_meta,'emotioNet_challenge_files_server')
    url_files = glob.glob(os.path.join(dir_url_files,'*.txt'))
    url_files.sort()

    
    args = []        
    for idx_url_file, url_file in enumerate(url_files):
        print 'On file %d of %d' %(idx_url_file,len(url_files)) 
        im = [line_curr.split('\t')[0] for line_curr in util.readLinesFromFile(url_file)]
        # out_files = [im_curr.replace(str_replace[0],str_replace[1]) for im_curr in im]


        for idx_im_curr,im_curr in enumerate(im):
            out_file_curr = im_curr.replace(str_replace[0],str_replace[1])
            if os.path.exists(out_file_curr):
                continue
            out_dir_curr = os.path.split(out_file_curr)[0]
            util.makedirs(out_dir_curr)
            args.append((im_curr, out_file_curr, idx_im_curr))

        print len(args)
        print url_file

    print len(args)

        # args = args[:5]
        # raw_input()
        # for arg in args:
        #     print arg
            # raw_input()
            # download_image(arg)
            # raw_input()
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(download_image,args)
    pool.close()
    pool.join()

    # urllib.urlcleanup()


    
        



def main():

    
    script_download_image()
    # idx_url_file,url_file,url_files, str_replace)


    return
    dir_meta = '../data/emotionet'
    dir_url_files = os.path.join(dir_meta,'emotioNet_challenge_files_server')
    url_files = glob.glob(os.path.join(dir_url_files,'*.txt'))
    all_im = []

    for url_file in url_files:
        im = util.readLinesFromFile(url_file)
        im = [line_curr.split('\t')[0] for line_curr in im]
        all_im = all_im+im

    print len(all_im)
    all_im = list(set(all_im))
    for im_curr in all_im:
        assert im_curr.startswith('http://cbcsnas01.ece.ohio-state.edu/EmotioNet/Images')

    print len(all_im)
    print all_im[0]
    

    print len(url_files)


if __name__=='__main__':
    main()