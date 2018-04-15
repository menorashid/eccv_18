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

def get_set_up(model_name, route_iter, pre_pend, strs_append, split_num, model_num, train_pre = None, test_file = None, au = False):

    # get_set_up(model_name, route_iter, pre_pend, strs_append, split_num, model_num, train_pre=train_pre, test_file = test_file ,au=True)


    out_dir_meta = os.path.join('../experiments',model_name+str(route_iter))
    out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
    # final_model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')  

    if train_pre is None:
        train_pre =  os.path.join('../data/ck_96','train_test_files')

    if test_file is None:
        test_file = os.path.join(train_pre,'test_'+str(split_num)+'.txt')
    else:
        test_file = os.path.join(train_pre,test_file)
        
    mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
    std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')
    data_transforms = {}
    data_transforms['val']= transforms.Compose([
            transforms.ToTensor(),
            lambda x: x*255.
            ])

    if au:
        test_data = dataset.Bp4d_Dataset_Mean_Std_Im(test_file, mean_file, std_file, resize= 96,transform = data_transforms['val'], binarize = True)
    else:
        test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    
    # if not au:
    if au:
        criterion = 'marginmulti'
    else:
        criterion = 'margin'
    test_params = dict(out_dir_train = out_dir_train,
                    model_num = model_num,
                    train_data = None,
                    test_data = test_data,
                    gpu_id = 0,
                    model_name = model_name,
                    batch_size_val = 32,
                    criterion = criterion,
                    au=au
                    )
    # num_iter = save_visualizations.save_routings(**test_params)    
    # print num_iter

    # # else:
    # #     print 'HERE'
    # #     raw_input()

    out_file_results = os.path.join(out_dir_train,'save_routings_single_batch_'+str(model_num))
    print out_file_results
    # # os.path.join('../scratch/ck_test','save_routings_single_batch_'+str(model_num))
    # # 
    # im_files = np.load(os.path.join(out_file_results,'ims_all.npy'))
    # captions = np.array(im_files)

    # im_files_new = []
    # captions_new = []
    # for r in range(im_files.shape[0]):
    #     caption_row = []
    #     im_row = []
    #     for c in range(im_files.shape[1]):
    #         file_curr = im_files[r,c]
    #         caption_row.append(os.path.split(file_curr)[1][:file_curr.rindex('.')])

    #         # print file_curr
    #         # print file_curr.replace(str_replace[0],str_replace[1])
    #         # print util.getRelPath(file_curr.replace(str_replace[0],str_replace[1]),dir_server)
    #         im_row.append(util.getRelPath(file_curr.replace(str_replace[0],str_replace[1]),dir_server))

    #         # im_files[r,c] = 
    #         # print im_files[r,c]
    #         # raw_input()
    #     im_files_new.append(im_row)
    #     captions_new.append(caption_row)

    # out_file_html = os.path.join(out_file_results,'visualizing_routing.html')
    # visualize.writeHTML(out_file_html,im_files_new,captions_new,96,96)
    # print out_file_html.replace(str_replace[0],str_replace[1]).replace(dir_server,click_str)


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

def get_routes_mat(dir_pre,route_pre,idx_keep,num_iter,batch_size):
    print idx_keep.shape,np.sum(idx_keep)
    idx_keep = np.where(idx_keep)[0]
    route_mat = []
    dict_loaded = {}
    for idx_curr in idx_keep:
        rel_bin = idx_curr//batch_size
        rel_idx = idx_curr - rel_bin*batch_size
        file_curr = os.path.join(dir_pre,route_pre+'_'+str(rel_bin)+'.npy')
        if file_curr in dict_loaded:
            route_all = dict_loaded[file_curr]
        else:
            dict_loaded[file_curr]=np.load(file_curr)
            route_all = dict_loaded[file_curr]

        # print idx_curr, rel_bin, rel_idx, file_curr
        # print route_all.shape
        route_curr = route_all[:,rel_idx:rel_idx+1,:,:]
        # print route_curr.shape
        # raw_input()
        
        route_mat.append(route_curr)

    print len(route_mat)
    print route_mat[0].shape
    route_mat = np.concatenate(route_mat,1)
    print route_mat.shape
    # raw_input()
    return route_mat
    # print idx_keep
    # raw_input()
    # print idx_keep.shape

def load_mats_au():
    dirs_rel = ['../experiments/khorrami_capsule_7_3_gray3/bp4d_train_test_files_110_gray_align_0_reconstruct_True_True_cropkhAugNoColor_marginmulti_False_wdecay_0_10_exp_0.96_350_1e-06_0.001_0.001_0.001_lossweights_1.0_1.0_None/save_routings_single_batch_9']
    
    out_dir = '../experiments/figures/bp4d_routing'
    util.makedirs(out_dir)
    num_iter = 62

    mats_names = ['labels','preds']
    # ,'routes_0','routes_1']
    mat_arrs = [[] for name in mats_names]

    for dir_curr in dirs_rel:
        # print dir_curr
        for idx_mat_name,mat_name in enumerate(mats_names):
            # print mat_name
            # arr_list = []
            for iter_curr in range(num_iter):
                # print iter_curr
                arr_curr_file = os.path.join(dir_curr,mat_name+'_'+str(iter_curr)+'.npy')
                arr_curr = np.load(arr_curr_file)
                # arr_list.append(arr_curr)
                mat_arrs[idx_mat_name].append(arr_curr)
    # mat_arrs = [np.concatenate(mat_arr,0) for mat_arr in mat_arrs]
    axis_combine = [0,0,1,1]
    mat_arrs = [np.concatenate(mat_arr,axis_curr) for mat_arr,axis_curr in zip(mat_arrs,axis_combine)]
    for mat_arr in mat_arrs:
        print mat_arr.shape 

    # raw_input()

    # print mat_arrs[0][:10],mat_arrs[1][:10]
    accuracy = np.sum(mat_arrs[0]==mat_arrs[1])/float(mat_arrs[0].size)
    print 'accuracy',accuracy

    # print mat_arrs
    routes_all = mat_arrs[2:]
    print len(routes_all)
    # raw_input()
    # num_emos = 
    range_emos = [11]
    # range(1,12)
    dict_routes = {}
    for label_curr in range_emos:
        idx_keep = np.logical_and(mat_arrs[0][:,label_curr]==1,mat_arrs[0][:,label_curr]==mat_arrs[1][:,label_curr])
        for route_num, route_pre in enumerate(['routes_0','routes_1']):
            routes_mat = get_routes_mat(dirs_rel[0], route_pre, idx_keep, num_iter, 32)
            dict_routes[(label_curr,route_num)]=routes_mat
        

    for label_curr in range_emos:
        for label_compare in range(12):
            for route_num, route_pre in enumerate(['routes_0','routes_1']):
                routes_rel = dict_routes[(label_curr,route_num)]
                routes_rel = routes_rel[label_compare]
                # raw_input()
            # for route_num,routes_0 in enumerate(routes_all):
                
            #     # routes_0 = mat_arrs[3]
            #     routes_0 = routes_0[label_compare,idx_keep,:,:]
                routes_rel = np.sum(routes_rel,2)
                print routes_rel.shape
                routes_rel = np.reshape(routes_rel,(routes_rel.shape[0],32,6,6))
                entropy = get_entropy(routes_rel)
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
    # out_dir = '../experiments/figures/ck_routing'
    # util.makedirs(out_dir)
    # out_file_table = os.path.join(out_dir,'ent_diff_table.txt')

    # str_file = []

    # num_emos = 8
    # emo_strs = ['Neutral','Anger', 'Contempt','Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']



    out_dir = '../experiments/figures/bp4d_routing'
    util.makedirs(out_dir)
    out_file_table = os.path.join(out_dir,'ent_diff_table.txt')

    str_file = []

    num_emos = 12
    aus = [1,2,4,6,7,10,12,14,15,17,23,24]
    emo_strs = ['AU '+str(num) for num in aus]
    
    labels = np.load(os.path.join(out_dir,'labels.py.npy'))
    preds = np.load(os.path.join(out_dir,'preds.py.npy'))

    print labels.shape, np.min(labels), np.max(labels)
    print preds.shape, np.min(preds), np.max(preds)

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

            # print label_arr[label_curr].shape
            idx_label = np.where(labels[:,label_curr]>0)[0]
            false_ents = []
            for label_false in range(num_emos):
                if label_false==label_curr:
                    continue
                label_arr_curr = label_arr[label_false]
                idx_label_curr = labels[labels[:,label_curr]>0,label_false]==0
                # print idx_label_curr.shape
                label_arr_keep = label_arr_curr[idx_label_curr]
                if label_arr_keep.size==0:
                    continue
                false_ents.append(np.mean(label_arr_keep))
            # print false_ents

            # print idx_label.shape
            # for x in label_arr:
            #     print x.shape
            # raw_input()
            true_ent = np.mean(label_arr[label_curr])
            # print true_ent
            # false_ents = [np.mean(label_arr[idx]) for idx in range(num_emos) if idx!=label_curr]
            mean_false_ents = np.mean(false_ents)
            diff = true_ent - mean_false_ents
            # print true_ent,np.mean(false_ents)
            # print label_curr, diff, true_ent,mean_false_ents
            str_print = '%s & %.3f & %.3f & %.3f' %(emo_strs[label_curr],diff,true_ent,mean_false_ents)
            print str_print
            str_file.append(str_print)
        print '___'
        str_file.append('___')

    util.writeFile(out_file_table,str_file)

            
            # raw_input()
           
def get_entropy_map():
    # out_dir = '../experiments/figures/ck_routing'
    # util.makedirs(out_dir)
    # out_file_table = os.path.join(out_dir,'ent_diff_table.txt')

    # str_file = []

    # num_emos = 8
    # emo_strs = ['Neutral','Anger', 'Contempt','Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

    out_dir = '../experiments/figures/bp4d_routing'
    util.makedirs(out_dir)
    out_file_table = os.path.join(out_dir,'ent_diff_table.txt')

    str_file = []

    num_emos = 12
    aus = [1,2,4,6,7,10,12,14,15,17,23,24]
    emo_strs = ['AU_'+str(num) for num in aus]

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
            
def get_class_variations(model_name, route_iter, pre_pend, strs_append, split_num, model_num, class_rel,type_exp,train_pre = None, test_file = None, au= False):
    # out_dir_meta = os.path.join('../experiments',model_name+str(route_iter))
    # out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
    # # final_model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')  

    # train_pre =  os.path.join('../data/ck_96','train_test_files')
    # test_file = os.path.join(train_pre,'test_'+str(split_num)+'.txt')
    # mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
    # std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')
    # data_transforms = {}
    # data_transforms['val']= transforms.Compose([
    #         transforms.ToTensor(),
    #         lambda x: x*255.
    #         ])

    # test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    # test_params = dict(out_dir_train = out_dir_train,
    #                 model_num = model_num,
    #                 train_data = None,
    #                 test_data = test_data,
    #                 gpu_id = 0,
    #                 model_name = model_name,
    #                 batch_size_val = None,
    #                 criterion = 'margin',
    #                 class_rel = class_rel
    #                 )
    


    out_dir_meta = os.path.join('../experiments',model_name+str(route_iter))
    out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
    # final_model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')  

    if train_pre is None:
        train_pre =  os.path.join('../data/ck_96','train_test_files')

    if test_file is None:
        test_file = os.path.join(train_pre,'test_'+str(split_num)+'.txt')
    else:
        test_file = os.path.join(train_pre,test_file)
    
    mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
    std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')
    data_transforms = {}
    data_transforms['val']= transforms.Compose([
            transforms.ToTensor(),
            lambda x: x*255.
            ])

    if au:
        test_data = dataset.Bp4d_Dataset_Mean_Std_Im(test_file, mean_file, std_file, resize= 96,transform = data_transforms['val'], binarize = True)
    else:
        test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    
    # if not au:
    if au:
        criterion = 'marginmulti'
    else:
        criterion = 'margin'
    test_params = dict(out_dir_train = out_dir_train,
                    model_num = model_num,
                    train_data = None,
                    test_data = test_data,
                    gpu_id = 0,
                    model_name = model_name,
                    batch_size_val = 128,
                    criterion = criterion,
                    au=au,
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
    elif type_exp ==2:
        
        save_visualizations.save_class_vary_attr(**test_params)
        out_file_results = os.path.join(out_dir_train,'save_class_vary_attr_single_batch_'+str(model_num)+'_'+str(class_rel))
        out_file_html = os.path.join(out_file_results,'visualizing_vary_attr_'+str(class_rel)+'.html')
    else:
        save_visualizations.save_class_vary_mag_class_rel(**test_params)
        # save_routings(**test_params)    
        out_file_results = os.path.join(out_dir_train,'save_class_vary_mag_single_batch_'+str(model_num)+'_'+str(class_rel))
        out_file_html = os.path.join(out_file_results,'visualizing_vary_mag_'+str(class_rel)+'.html')
        # os.path.join('../scratch/ck_test','save_routings_single_batch_'+str(model_num))


    return
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



def script_visualizing_primary_caps():
    model_name = 'khorrami_capsule_7_3_bigclass'
    route_iter = 3
    pre_pend = 'ck_96_train_test_files_'
    strs_append = '_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_exp_0.96_350_1e-06_0.001_0.001_0.001'
    model_num = 599
    split_num = 4
    
    out_dir_meta = os.path.join('../experiments',model_name+str(route_iter))
    out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
    train_pre =  os.path.join('../data/ck_96','train_test_files')
    test_file =  os.path.join(train_pre,'train_'+str(split_num)+'.txt')
    
    mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
    std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')
    data_transforms = {}
    data_transforms['val']= transforms.Compose([
            transforms.ToTensor(),
            lambda x: x*255.
            ])

    test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    au = False
    class_rel = 0
    criterion = 'margin'
    test_params = dict(out_dir_train = out_dir_train,
                    model_num = model_num,
                    train_data = None,
                    test_data = test_data,
                    gpu_id = 0,
                    model_name = model_name,
                    batch_size_val = 128,
                    criterion = criterion,
                    au=au,
                    class_rel = class_rel
                    )


    save_visualizations.save_primary_caps(**test_params)


def main():
    # get_entropy_map()
    # get_entropy_table()
    # load_mats_au()

    script_visualizing_primary_caps()

    return

    model_name = 'khorrami_capsule_7_3_gray'
    route_iter = 3
    pre_pend =  'bp4d_train_test_files_110_gray_align_'
    strs_append = '_reconstruct_True_True_cropkhAugNoColor_marginmulti_False_wdecay_0_10_exp_0.96_350_1e-06_0.001_0.001_0.001_lossweights_1.0_1.0_None'
    model_num = 9
    split_num = 0
    train_pre = os.path.join('../data/bp4d','train_test_files_110_gray_align')
    test_file = 'test_0_best_results.txt'

    # get_set_up(model_name, route_iter, pre_pend, strs_append, split_num, model_num, train_pre=train_pre, test_file = test_file ,au=True)
    type_exp = 3

    for class_rel in range(0,12):
        get_class_variations(model_name, route_iter, pre_pend, strs_append, split_num, model_num, class_rel,type_exp, train_pre = train_pre, test_file=test_file,au=True)

    return
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