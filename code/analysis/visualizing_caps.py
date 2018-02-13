import sys
sys.path.append('./')
from helpers import util, visualize
import os
import numpy as np
import scipy.misc

def make_html_test_data():   
    replace_str = ['..','/disk3/maheen_data/eccv_18']
    dir_server = '/disk3'

    # out_dir_results = '../experiments/dynamic_capsules/ck_0_108_exp_0.001/out_caps_model_test_107'
    # train_file = '../data/ck_96/train_test_files/test_0.txt'

    out_dir_results = '../experiments/dynamic_capsules/with_recon_ck_tanh0_108_exp_0.001/out_caps_model_test_107'
    train_file = '../data/ck_96/train_test_files/test_0.txt'
    # reconstruct = True

    out_all = np.load(os.path.join(out_dir_results, 'out_all.npy'))
    predictions = np.load(os.path.join(out_dir_results, 'predictions.npy'))
    labels_all = np.load(os.path.join(out_dir_results, 'labels_all.npy'))
    caps_all = np.load(os.path.join(out_dir_results, 'caps_all.npy'))

    # for every class have an html with every row going most to least

    lines = util.readLinesFromFile(train_file)
    lines = [line.split(' ') for line in lines]
    im_files = np.array([line[0] for line in lines])
    labels_file = np.array([int(line[1]) for line in lines])
    assert np.all(labels_file==labels_all)

    for emotion in list(np.unique(labels_all))+['all']:
        if emotion=='all':
            out_file_html = os.path.join(out_dir_results,str(emotion)+'.html')
            bin_emotion = labels_all==predictions
            caps_rel =[caps_all[im_num,labels_all[im_num],:][np.newaxis,:] for im_num in range(labels_all.shape[0]) if labels_all[im_num]==predictions[im_num]]
            caps_rel = np.concatenate(caps_rel,0)
            print caps_rel.shape
            # raw_input()
            # caps_rel = caps_rel[:,labels_all[bin_emotion],:]


        else:
            out_file_html = os.path.join(out_dir_results,str(emotion)+'.html')
            bin_emotion = np.logical_and(labels_all==emotion,labels_all==predictions)
            caps_rel = caps_all[bin_emotion,emotion,:]

        print caps_rel.shape
        im_files_rel = im_files[bin_emotion]
        print im_files_rel.shape
        # raw_input()

        im_files_html = []
        captions_html = []
        for dimension in range(caps_rel.shape[1]):
            idx_sort = np.argsort(caps_rel[:,dimension])[::-1]
            
            im_files_sorted = im_files_rel[idx_sort]
            caps_sorted = caps_rel[idx_sort,dimension]

            print idx_sort.shape
            im_files_curr = [util.getRelPath(file_curr.replace(replace_str[0],replace_str[1]),dir_server) for file_curr in im_files_sorted]
            captions_curr = [str(dimension)+' '+os.path.split(file_curr)[1]+' '+str(caps_sorted[idx_file_curr]) for idx_file_curr,file_curr in enumerate(im_files_sorted)] 
            im_files_html.append(im_files_curr)
            captions_html.append(captions_curr)

        visualize.writeHTML(out_file_html,im_files_html,captions_html,96,96)

def view_reconstruction():
    replace_str = ['..','/disk3/maheen_data/eccv_18']
    dir_server = '/disk3'

    out_dir_results = '../experiments/dynamic_capsules/with_recon_ck_notanh_nosig_fixtrain_0_108_exp_0.001/out_caps_model_test_107'
    out_dir_recons = os.path.join(out_dir_results,'recon_im').replace(replace_str[0],replace_str[1])
    util.mkdir(out_dir_recons)
    train_file = '../data/ck_96/train_test_files/test_0.txt'
    mean_file = '../data/ck_96/train_test_files/train_0_mean.png'
    std_file = '../data/ck_96/train_test_files/train_0_std.png'
    im_size = 28

    mean_im = scipy.misc.imresize(scipy.misc.imread(mean_file),(im_size,im_size)).astype(np.float32)
    std_im = scipy.misc.imresize(scipy.misc.imread(std_file),(im_size,im_size)).astype(np.float32)
    std_im[std_im==0]=1.

    out_all = np.load(os.path.join(out_dir_results, 'out_all.npy'))
    predictions = np.load(os.path.join(out_dir_results, 'predictions.npy'))
    labels_all = np.load(os.path.join(out_dir_results, 'labels_all.npy'))
    caps_all = np.load(os.path.join(out_dir_results, 'caps_all.npy'))
    recons_all = np.load(os.path.join(out_dir_results, 'recons_all.npy'))   
    
    lines = util.readLinesFromFile(train_file)

    im_files_html = []
    captions_html = []
    out_file_html = os.path.join(out_dir_results,'recons_viz.html')

    for idx_line, line in enumerate(lines):
        im_file,label = line.split(' ')
        im_file = im_file.replace(replace_str[0],replace_str[1])
        recons = recons_all[idx_line][0]
        recons = (recons*std_im)+mean_im
        out_file = os.path.join(out_dir_recons,os.path.split(im_file)[1])
        scipy.misc.imsave(out_file,recons)

        im_files_html.append([util.getRelPath(file_curr,dir_server) for file_curr in [im_file,out_file]])
        captions_html.append(['true '+label,'recon '+str(predictions[idx_line])])

    visualize.writeHTML(out_file_html,im_files_html,captions_html,28,28)     


def main():
    replace_str = ['..','/disk3/maheen_data/eccv_18']
    dir_server = '/disk3'
    out_dir_results = '../experiments/dynamic_capsules/with_recon_ck_notanh_nosig_fixtrain_0_108_exp_0.001/vary_a_batch_107'.replace(replace_str[0],replace_str[1])
    train_file = '../data/ck_96/train_test_files/test_0.txt'
    
    lines = util.readLinesFromFile(train_file)
    perturb_vals = list(np.arange(-0.25,0.3,0.05))
    dims = range(16)

    batch_size_val = 128
    im_files_html = []
    captions_html = []
    out_file_html = os.path.join(out_dir_results,'recons_viz.html')

    predictions = np.load(os.path.join(out_dir_results.replace('vary_a_batch_107','out_caps_model_test_107'),'predictions.npy'))

    for dim in dims:
        out_file_html = os.path.join(out_dir_results,'dim_'+str(dim)+'.html')
        im_html =[]
        captions_html = []

        for im_num in range(batch_size_val):
            im_row = []
            caption_row = []

            im_file, label =lines[im_num].split(' ')
            label_pred = predictions[im_num]
            im_row.append(util.getRelPath(im_file.replace(replace_str[0],replace_str[1]),dir_server))
            caption_row.append(str(im_num)+' '+label)

        # other cases
            for dim_curr, perturb_val in [(-1,-2),(-1,-1)]:
                folder_curr = '%d_%.2f'%(dim_curr,perturb_val)
                im_curr = os.path.join(out_dir_results,folder_curr,str(im_num)+'.jpg')
                im_curr = util.getRelPath(im_curr,dir_server)
                im_row.append(im_curr)
                caption_row.append(' '.join([str(im_num),label if dim_curr else str(label_pred)]))

            for perturb_val in perturb_vals:
                folder_curr = '%d_%.2f'%(dim,perturb_val)
                im_curr = os.path.join(out_dir_results,folder_curr,str(im_num)+'.jpg')
                im_curr = util.getRelPath(im_curr,dir_server)
                im_row.append(im_curr)
                # if im_num==0:
                #     caption_row.append('%.2f'%perturb_val)
                # else:
                caption_row.append('')

            im_html.append(im_row)    
            captions_html.append(caption_row)
        visualize.writeHTML(out_file_html,im_html,captions_html,28,28)









if __name__=='__main__':
    main()
