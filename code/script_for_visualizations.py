from train_test_caps import *
from torchvision import datasets, transforms
import models

import os
from helpers import util,visualize,augmenters
import random
import dataset
import numpy as np
import torch
from analysis import getting_accuracy
from helpers import util,visualize,augmenters
import save_visualizations

dir_server = '/disk3'
str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
click_str = 'http://vision3.idav.ucdavis.edu:1000'


def get_ck_16_dirs():

    model_name = 'khorrami_capsule_7_3'
    route_iter = 3
    pre_pend = 'ck_96_'
    strs_append = '_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001'
    model_num = 599    
    out_file_results_all = []
    for split_num in range(10):
        out_dir_meta = os.path.join('../experiments',model_name+str(route_iter))
        out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
        out_file_results = os.path.join(out_dir_train,'save_routings_single_batch_'+str(model_num))
        out_file_results_all.append(out_file_results)
    return out_file_results_all

def get_set_up(model_name, route_iter, pre_pend, strs_append, split_num, model_num):
    out_dir_meta = os.path.join('../experiments',model_name+str(route_iter))
    out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
    # final_model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')  

    train_pre =  os.path.join('../data/ck_96','train_test_files')
    test_file = os.path.join(train_pre,'test_'+str(split_num)+'.txt')
    mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
    std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')
    data_transforms = {}
    data_transforms['val']= transforms.Compose([
            transforms.ToTensor(),
            lambda x: x*255.
            ])

    test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    test_params = dict(out_dir_train = out_dir_train,
                    model_num = model_num,
                    train_data = None,
                    test_data = test_data,
                    gpu_id = 0,
                    model_name = model_name,
                    batch_size_val = None,
                    criterion = 'margin',
                    )
    save_visualizations.save_routings(**test_params)    
    out_file_results = os.path.join(out_dir_train,'save_routings_single_batch_'+str(model_num))
    # os.path.join('../scratch/ck_test','save_routings_single_batch_'+str(model_num))
    # 
    im_files = np.load(os.path.join(out_file_results,'ims_all.npy'))
    captions = np.array(im_files)

    im_files_new = []
    captions_new = []
    for r in range(im_files.shape[0]):
        caption_row = []
        im_row = []
        for c in range(im_files.shape[1]):
            file_curr = im_files[r,c]
            caption_row.append(os.path.split(file_curr)[1][:file_curr.rindex('.')])

            # print file_curr
            # print file_curr.replace(str_replace[0],str_replace[1])
            # print util.getRelPath(file_curr.replace(str_replace[0],str_replace[1]),dir_server)
            im_row.append(util.getRelPath(file_curr.replace(str_replace[0],str_replace[1]),dir_server))

            # im_files[r,c] = 
            # print im_files[r,c]
            # raw_input()
        im_files_new.append(im_row)
        captions_new.append(caption_row)

    out_file_html = os.path.join(out_file_results,'visualizing_routing.html')
    visualize.writeHTML(out_file_html,im_files_new,captions_new,96,96)
    print out_file_html.replace(str_replace[0],str_replace[1]).replace(dir_server,click_str)


def save_routes():

    model_name = 'khorrami_capsule_7_3_bigclass'
    route_iter = 3
    dir_exp_pre = 'ck_96_train_test_files_'
    split_num = 0
    dir_exp_post = '_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_exp_0.96_350_1e-06_0.001_0.001_0.001'
    model_num = 599

    model_name = 'khorrami_capsule_7_3'
    route_iter = 3
    dir_exp_pre = 'ck_96_'
    dir_exp_post = '_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001'
    model_num = 599    
    for split_num in range(10):
        get_set_up(model_name, route_iter, dir_exp_pre, dir_exp_post, split_num, model_num)
        break
        print 'hello'

def get_entropy(volume):
    entropy = np.zeros((volume.shape[0],volume.shape[2],volume.shape[3]))
    for im_num in range(volume.shape[0]):
        for r in range(volume.shape[2]):
            for c in range(volume.shape[3]):
                vec = volume[im_num,:,r,c]
                entropy[im_num,r,c] = scipy.stats.entropy(vec)

    # entropy = np.mean(entropy,0)
    return entropy


def load_mats():
    dirs_rel = get_ck_16_dirs()
    
    out_dir = '../experiments/figures/ck_routing'
    util.makedirs(out_dir)

    mats_names = ['labels','preds','routes_0','routes_1']
    mat_arrs = [[] for name in mats_names]
    for dir_curr in dirs_rel:
        # print dir_curr
        for idx_mat_name,mat_name in enumerate(mats_names):
            arr_curr_file = os.path.join(dir_curr,mat_name+'.npy')
            arr_curr = np.load(arr_curr_file)
            mat_arrs[idx_mat_name].append(arr_curr)
    # mat_arrs = [np.concatenate(mat_arr,0) for mat_arr in mat_arrs]
    axis_combine = [0,0,1,1]
    mat_arrs = [np.concatenate(mat_arr,axis_curr) for mat_arr,axis_curr in zip(mat_arrs,axis_combine)]
    for mat_arr in mat_arrs:
        print mat_arr.shape 

    # print mat_arrs[0][:10],mat_arrs[1][:10]
    accuracy = np.sum(mat_arrs[0]==mat_arrs[1])/float(mat_arrs[0].size)
    print 'accuracy',accuracy

    # print mat_arrs
    routes_all = mat_arrs[2:]
    print len(routes_all)
    # raw_input()
    num_emos = 8

    for label_curr in range(num_emos):
        for label_compare in range(num_emos):
            for route_num,routes_0 in enumerate(routes_all):
                idx_keep = np.logical_and(mat_arrs[0]==label_curr,mat_arrs[0]==mat_arrs[1])
                # routes_0 = mat_arrs[3]
                routes_0 = routes_0[label_compare,idx_keep,:,:]
                routes_0 = np.sum(routes_0,2)
                routes_0 = np.reshape(routes_0,(routes_0.shape[0],32,6,6))
                entropy = get_entropy(routes_0)
                print entropy.shape
                file_name = [label_curr,label_compare,route_num]
                out_file = os.path.join(out_dir,'_'.join([str(val) for val in file_name])+'.npy')
                print out_file
                # raw_input()
                np.save(out_file,entropy)
    
    np.save(os.path.join(out_dir,'labels.py'),  mat_arrs[0])
    np.save(os.path.join(out_dir,'preds.py'),  mat_arrs[1])
        # print routes_0.shape, np.min(routes_0),np.max(routes_0)


def get_entropy_table():
    out_dir = '../experiments/figures/ck_routing'
    util.makedirs(out_dir)
    out_file_table = os.path.join(out_dir,'ent_diff_table.txt')

    str_file = []

    num_emos = 8
    emo_strs = ['Neutral','Anger', 'Contempt','Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

    num_routes = 2
    for route_num in range(num_routes):
        print route_num
        str_file.append(str(route_num))
        for label_curr in range(num_emos):
            label_arr = []
            for label_compare in range(num_emos):
                file_name = [label_curr,label_compare,route_num]
                ent_file = os.path.join(out_dir,'_'.join([str(val) for val in file_name])+'.npy')
                ent_curr = np.load(ent_file)
                label_arr.append(ent_curr)


            true_ent = np.mean(label_arr[label_curr])
            # print true_ent
            false_ents = [np.mean(label_arr[idx]) for idx in range(num_emos) if idx!=label_curr]
            mean_false_ents = np.mean(false_ents)
            diff = true_ent - mean_false_ents
            # print true_ent,np.mean(false_ents)
            # print label_curr, diff, true_ent,mean_false_ents
            str_print = '%s & %.2f & %.2f & %.2f' %(emo_strs[label_curr],diff,true_ent,mean_false_ents)
            print str_print
            str_file.append(str_print)
        print '___'
        str_file.append('___')

    util.writeFile(out_file_table,str_file)

            
            # raw_input()



                
def get_entropy_map():
    out_dir = '../experiments/figures/ck_routing'
    util.makedirs(out_dir)
    out_file_table = os.path.join(out_dir,'ent_diff_table.txt')

    str_file = []

    num_emos = 8
    emo_strs = ['Neutral','Anger', 'Contempt','Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

    num_routes = 2
    for route_num in range(num_routes):
        print route_num
        str_file.append(str(route_num))
        for label_curr in range(num_emos):
            label_arr = []
            for label_compare in range(num_emos):
                file_name = [label_curr,label_compare,route_num]
                ent_file = os.path.join(out_dir,'_'.join([str(val) for val in file_name])+'.npy')
                ent_curr = np.load(ent_file)
                label_arr.append(ent_curr)


            # true_ent = np.mean(label_arr[label_curr],0)
            # print true_ent
            all_ents = [np.mean(label_arr[idx],0) for idx in range(num_emos)]
            catted = np.concatenate(all_ents,0)
            min_val = np.min(catted)
            max_val = np.max(catted)
            for idx_ent_curr, ent_curr in enumerate(all_ents):
                out_file_curr = os.path.join(out_dir,'_'.join(str(val) for val in [label_curr,idx_ent_curr,route_num,route_num])+'.png')
                title = emo_strs[label_curr]+' '+emo_strs[idx_ent_curr]

                visualize.plot_colored_mats(out_file_curr,ent_curr,min_val,max_val, title=title)

            

    visualize.writeHTMLForFolder(out_dir,'.png')
            


def get_class_variations(model_name, route_iter, pre_pend, strs_append, split_num, model_num, class_rel,type_exp):
    out_dir_meta = os.path.join('../experiments',model_name+str(route_iter))
    out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
    # final_model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')  

    train_pre =  os.path.join('../data/ck_96','train_test_files')
    test_file = os.path.join(train_pre,'test_'+str(split_num)+'.txt')
    mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
    std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')
    data_transforms = {}
    data_transforms['val']= transforms.Compose([
            transforms.ToTensor(),
            lambda x: x*255.
            ])

    test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    test_params = dict(out_dir_train = out_dir_train,
                    model_num = model_num,
                    train_data = None,
                    test_data = test_data,
                    gpu_id = 0,
                    model_name = model_name,
                    batch_size_val = None,
                    criterion = 'margin',
                    class_rel = class_rel
                    )
    
    if type_exp ==0 :
        save_visualizations.save_class_as_other(**test_params)
        # save_routings(**test_params)    
        out_file_results = os.path.join(out_dir_train,'save_class_as_other_single_batch_'+str(model_num))
        out_file_html = os.path.join(out_file_results,'visualizing_class_variations_'+str(class_rel)+'.html')
        # os.path.join('../scratch/ck_test','save_routings_single_batch_'+str(model_num))
    elif type_exp==1:
        save_visualizations.save_class_vary_mag(**test_params)
        # save_routings(**test_params)    
        out_file_results = os.path.join(out_dir_train,'save_class_vary_mag_single_batch_'+str(model_num))
        out_file_html = os.path.join(out_file_results,'visualizing_vary_mag_'+str(class_rel)+'.html')
        # os.path.join('../scratch/ck_test','save_routings_single_batch_'+str(model_num))
    else:
        
        save_visualizations.save_class_vary_attr(**test_params)
        out_file_results = os.path.join(out_dir_train,'save_class_vary_attr_single_batch_'+str(model_num))
        out_file_html = os.path.join(out_file_results,'visualizing_vary_attr_'+str(class_rel)+'.html')
        
        # save_visualizations.save_class_vary_mag(**test_params)
        # # save_routings(**test_params)    
        # out_file_results = os.path.join(out_dir_train,'save_class_as_other_single_batch_'+str(model_num))
        # os.path.join('../scratch/ck_test','save_routings_single_batch_'+str(model_num))

    im_files = np.load(os.path.join(out_file_results,'ims_all.npy'))
    captions = np.array(im_files)

    im_files_new = []
    captions_new = []
    for r in range(im_files.shape[0]):
        caption_row = []
        im_row = []
        for c in range(im_files.shape[1]):
            file_curr = im_files[r,c]
            caption_row.append(os.path.split(file_curr)[1][:file_curr.rindex('.')])

            # print file_curr
            # print file_curr.replace(str_replace[0],str_replace[1])
            # print util.getRelPath(file_curr.replace(str_replace[0],str_replace[1]),dir_server)
            im_row.append(util.getRelPath(file_curr.replace(str_replace[0],str_replace[1]),dir_server))

            # im_files[r,c] = 
            # print im_files[r,c]
            # raw_input()
        im_files_new.append(im_row)
        captions_new.append(caption_row)

    
    visualize.writeHTML(out_file_html,im_files_new,captions_new,96,96)
    print out_file_html.replace(str_replace[0],str_replace[1]).replace(dir_server,click_str)


def main():
    
    model_name = 'khorrami_capsule_7_3_bigclass'
    route_iter = 3
    pre_pend = 'ck_96_train_test_files_'
    strs_append = '_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_exp_0.96_350_1e-06_0.001_0.001_0.001'
    model_num = 599
    split_num = 4
    # class_rel = 0
    type_exp = 2
    for class_rel in range(8):
        get_class_variations(model_name, route_iter, pre_pend, strs_append, split_num, model_num,class_rel,type_exp = type_exp)
        



    # get_entropy_map()
    # get_entropy_table()
    # save_routes()
    # load_mats()







if __name__=='__main__':
    main()