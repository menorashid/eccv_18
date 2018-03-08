import sys
sys.path.append('./')
from helpers import util, visualize
import os
import numpy as np
import scipy.misc

dir_server = '/disk3'
str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
click_str = 'http://vision3.idav.ucdavis.edu:1000'

def make_html_recon(out_dir_results,mean_file,std_file ):
    
    out_dir_results = out_dir_results.replace(str_replace[0],str_replace[1])

    recon_all_pred = np.load(os.path.join(out_dir_results,'recon_all.npy')).squeeze()
    recon_all_gt = np.load(os.path.join(out_dir_results,'recon_all_gt.npy')).squeeze()
    im_org = np.load(os.path.join(out_dir_results,'im_org.npy')).squeeze()

    out_dir_im = os.path.join(out_dir_results,'im_recon')
    util.mkdir(out_dir_im)

    std_im = scipy.misc.imread(std_file).astype(np.float32)
    # print np.min(std_im),np.max(std_im)
    mean_im = scipy.misc.imread(mean_file).astype(np.float32)
    # print np.min(mean_im),np.max(mean_im)
    # raw_input()

    out_file_html = os.path.join(out_dir_results,'im_recon.html')
    ims_html = []
    captions_html = []

    strs_pre = ['org','gt','pred']
    for idx in range(im_org.shape[0]):
        im_row = []
        caption_row = []
        for str_curr,np_curr in zip(strs_pre,[im_org,recon_all_gt,recon_all_pred]):
            im_curr = np_curr[idx]
            # print np.min(im_curr),np.max(im_curr)
            im_curr = (im_curr*std_im)+mean_im
            # im_curr = im_curr-np.min(im_curr)
            # im_curr = im_curr/np.max(im_curr)*255
            out_file_curr = os.path.join(out_dir_im,str_curr+'_'+str(idx)+'.jpg')
            

            scipy.misc.imsave(out_file_curr,im_curr)

            im_row.append(util.getRelPath(out_file_curr,dir_server))
            caption_row.append(' '.join([str(idx), str_curr]))

        ims_html.append(im_row)
        captions_html.append(caption_row)
        
        

    visualize.writeHTML(out_file_html,ims_html,captions_html,96,96)
    print out_file_html.replace(dir_server,click_str)

        
        



def main():
    split_nums = [2,9]
    data_dir_meta = '../data/ck_96/train_test_files'
    out_dir_results = '../experiments/khorrami_capsule_7_33/ck_96_0_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001/results_model_100'

    type_models = ['khorrami_capsule_7_3']
    # _bigrecon','khorrami_capsule_7_3_bigclass']

    for split_curr in split_nums:
        for model_curr in type_models:
            out_dir_results = os.path.join('../experiments',model_curr+'3','ck_96_'+str(split_curr)+'_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001/results_model_599')
            out_dir_results_train = out_dir_results+'_train'
            mean_file = os.path.join(data_dir_meta, 'train_'+str(split_curr)+'_mean.png')
            std_file = os.path.join(data_dir_meta, 'train_'+str(split_curr)+'_std.png')
            make_html_recon(out_dir_results,mean_file,std_file)
            make_html_recon(out_dir_results_train,mean_file,std_file)



if __name__=='__main__':
    main()

