import sys
sys.path.append('./')
from helpers import util, visualize
import os
import numpy as np
import scipy.misc
import glob

dir_server = '/disk3'
str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
click_str = 'http://vision3.idav.ucdavis.edu:1000'


def make_html_recon_active_thresh(out_dir_results, mean_file, std_file, au_arr, resize = None, thresh_active= 6, thresh_pred = 0.5):

    label_pre = 'labels_all'
    pred_pre = 'predictions'
    recon_gt_pre = 'recon_all_gt'
    im_org_pre = 'im_org'
    recon_pred_pre = 'recon_all'
    au_arr = np.array(au_arr)[np.newaxis,:]
    
    out_dir_results = out_dir_results.replace(str_replace[0],str_replace[1])

    mean = scipy.misc.imread(mean_file)
    std = scipy.misc.imread(std_file)
    if resize is not None:
        mean = scipy.misc.imresize(mean,(resize,resize))
        std = scipy.misc.imresize(std,(resize,resize))
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    num_files = glob.glob(os.path.join(out_dir_results, label_pre+'*.npy'))
    num_files = [int(file_curr[file_curr.rindex('_')+1:file_curr.rindex('.')]) for file_curr in num_files]
    num_files.sort()

    strs_load = [im_org_pre, recon_gt_pre, recon_pred_pre]
    strs_pre = ['org', 'gt', 'pred']
    ims_all = [[],[],[]]
    label_all = []
    pred_all = []

    for num_curr in num_files[:10]:
        label_file = os.path.join(out_dir_results,'_'.join([label_pre,str(num_curr)+'.npy']))
        pred_file = os.path.join(out_dir_results,'_'.join([pred_pre,str(num_curr)+'.npy']))
        label_curr = np.load(label_file)

        pred_curr = np.load(pred_file)
        bin_keep = np.sum(label_curr,1)>=thresh_active
        if np.sum(bin_keep)==0:
            continue
        print bin_keep.shape, np.sum(bin_keep)
        # bin_keep = np.sum(pred_curr>=0.95,1)>0
        
        label_all.append(label_curr[bin_keep])
        pred_all.append(pred_curr[bin_keep])

        for idx_str_curr, str_curr in enumerate(strs_load):
            file_curr = os.path.join(out_dir_results,'_'.join([str_curr,str(num_curr)+'.npy']))
            im_curr = np.load(file_curr)
            im_rel = im_curr[bin_keep]
            ims_all[idx_str_curr].append(im_rel)

    out_dir_im = os.path.join(out_dir_results,'im_to_disp_'+str(thresh_active))
    util.mkdir(out_dir_im)
    out_file_html = out_dir_im+'.html' 
    ims_all = [np.concatenate(arr_curr,0) for arr_curr in ims_all]
    label_all = np.concatenate(label_all,0)
    pred_all = np.concatenate(pred_all,0)

    print label_all.shape
    label_all = label_all*au_arr

    pred_all[pred_all<=thresh_pred]=0
    pred_all[pred_all>thresh_pred]=1
    print pred_all.shape, au_arr.shape
    pred_all = pred_all.astype(int)
    pred_all = pred_all*au_arr

    ims_all = [ims_curr.squeeze()*std[np.newaxis,:,:]+mean[np.newaxis,:,:] for ims_curr in ims_all]
    # ims_all = [ims_curr.squeeze() for ims_curr in ims_all]
    # ims_all[0] = ims_all[0]*std[np.newaxis,:,:]+mean[np.newaxis,:,:] 

    # print ims_all[0].shape, mean.shape, std.shape
    caption_arrs = [None,label_all, pred_all]
    save_im_make_html(strs_pre, out_file_html, out_dir_im, ims_all, caption_arrs)



def save_im_make_html(strs_pre, out_file_html, out_dir_im,  ims_all, caption_arrs):
    
    
    ims_html = []
    captions_html = []
    for idx_row in range(ims_all[0].shape[0]):
        im_row = []
        caption_row = []
        for idx_col, str_pre in enumerate(strs_pre):
            im_curr = ims_all[idx_col][idx_row]
            out_file_curr = os.path.join(out_dir_im, str_pre+'_'+str(idx_row)+'.jpg')
            # im_curr = im_curr+np.min(im_curr)
            print np.min(im_curr), np.max(im_curr)
            # im_curr = im_curr.astype(np.uint8)
            scipy.misc.imsave(out_file_curr,im_curr)

            caption_curr = [str(idx_col),str_pre]
            
            if caption_arrs[idx_col] is not None:
                caption_curr = caption_curr+[str(val) for val in caption_arrs[idx_col][idx_row] if val!=0]
            
            caption_curr = ' '.join(caption_curr)
            caption_row.append(caption_curr)

            im_row.append(util.getRelPath(out_file_curr,dir_server))
            # raw_input()

        ims_html.append(im_row)
        captions_html.append(caption_row)
        
        

    visualize.writeHTML(out_file_html,ims_html,captions_html,96,96)
    print out_file_html.replace(dir_server,click_str)

            






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

