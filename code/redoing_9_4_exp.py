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



def train_khorrami_aug(wdecay,lr,route_iter,folds=[4,9],model_name='vgg_capsule_disfa',epoch_stuff=[30,60],res=False, class_weights = False, reconstruct = False, oulu = False, meta_data_dir = None,loss_weights = None, exp = False, non_peak = False):
    out_dirs = []

    out_dir_meta = '../experiments/'+model_name+str(route_iter)
    num_epochs = epoch_stuff[1]
    epoch_start = 0
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    # dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr
    im_resize = 110
    im_size = 96
    save_after = 50
    if non_peak:
        type_data = 'train_test_files_non_peak_one_third'; n_classes = 8;
        train_pre = os.path.join('../data/ck_96',type_data)
        test_pre =  os.path.join('../data/ck_96','train_test_files')
    else:
        type_data = 'train_test_files'; n_classes = 8;
        train_pre = os.path.join('../data/ck_96',type_data)
        test_pre =  os.path.join('../data/ck_96',type_data)

    if oulu:
        type_data = 'three_im_no_neutral_just_strong_False'; n_classes = 6;
    criterion = 'margin'
    criterion_str = criterion

    # criterion = nn.CrossEntropyLoss()
    # criterion_str = 'crossentropy'
    
    init = False
    strs_append_list = ['reconstruct',reconstruct,class_weights,'all_aug',criterion_str,init,'wdecay',wdecay,num_epochs]+dec_after+lr

    if loss_weights is not None:
        strs_append_list = strs_append_list     +['lossweights']+loss_weights
    strs_append = '_'+'_'.join([str(val) for val in strs_append_list])
    
    if oulu:
        pre_pend = 'oulu_96_'+meta_data_dir+'_'
    else:
        pre_pend = 'ck_96_'+type_data+'_'
    
    lr_p=lr[:]
    for split_num in folds:
        
        if res:

            strs_appendc = '_'+'_'.join([str(val) for val in ['reconstruct',reconstruct,True,'all_aug',criterion_str,init,'wdecay',wdecay,600,'step',600,0.1]+lr_p])
            out_dir_train = os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_appendc)
            model_file = os.path.join(out_dir_train,'model_599.pt')
            epoch_start = 600
            lr =[0.1*lr_curr for lr_curr in lr_p]
        else:
            model_file = None    


        margin_params = None
        
        out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
        final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
        if os.path.exists(final_model_file):
            print 'skipping',final_model_file
            # raw_input()
            continue 
        else:
            print 'not skipping', final_model_file
            # raw_input()
            # continue

        if not oulu:
            # train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
            # test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
            # mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
            # std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'

            train_file = os.path.join(train_pre,'train_'+str(split_num)+'.txt')
            test_file = os.path.join(test_pre,'test_'+str(split_num)+'.txt')
            mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
            std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')

        else:
            train_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'.txt')
            test_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'test_'+str(split_num)+'.txt')
            mean_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'_mean.png')
            std_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'_std.png')

        mean_im = scipy.misc.imread(mean_file).astype(np.float32)
        std_im = scipy.misc.imread(std_file).astype(np.float32)

        class_weights = util.get_class_weights(util.readLinesFromFile(train_file))

        # print std_im.shape
        # print np.min(std_im),np.max(std_im)
        # raw_input()

        list_of_to_dos = ['flip','rotate','scale_translate', 'pixel_augment']
        
        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            lambda x: augmenters.augment_image(x,list_of_to_dos,mean_im,std_im,im_size),
            transforms.ToTensor(),
            lambda x: x*255.
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToTensor(),
            lambda x: x*255.
            ])

        # train_data = dataset.CK_96_Dataset_Just_Mean(train_file, mean_file, data_transforms['train'])
        # test_data = dataset.CK_96_Dataset_Just_Mean(test_file, mean_file, data_transforms['val'])

        print train_file
        print test_file
        print std_file
        print mean_file
        # raw_input()

        train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
        train_data_no_t = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['val'])
        test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
        
        network_params = dict(n_classes=n_classes,pool_type='max',r=route_iter,init=init,class_weights = class_weights, reconstruct = reconstruct,loss_weights = loss_weights)
        # if lr[0]==0:
        batch_size = 128
        batch_size_val = 128
        # else:
        #     batch_size = 32
        #     batch_size_val = 16

        util.makedirs(out_dir_train)
        
        train_params = dict(out_dir_train = out_dir_train,
                    train_data = train_data,
                    test_data = test_data,
                    batch_size = batch_size,
                    batch_size_val = batch_size_val,
                    num_epochs = num_epochs,
                    save_after = save_after,
                    disp_after = 1,
                    plot_after = 100,
                    test_after = 1,
                    lr = lr,
                    dec_after = dec_after, 
                    model_name = model_name,
                    criterion = criterion,
                    gpu_id = 0,
                    num_workers = 2,
                    model_file = model_file,
                    epoch_start = epoch_start,
                    margin_params = margin_params,
                    network_params = network_params,
                    weight_decay=wdecay)
        test_params = dict(out_dir_train = out_dir_train,
                    model_num = num_epochs-1, 
                    train_data = train_data,
                    test_data = test_data,
                    gpu_id = 0,
                    model_name = model_name,
                    batch_size_val = batch_size_val,
                    criterion = criterion,
                    margin_params = margin_params,
                    network_params = network_params)
        test_params_train = dict(**test_params)
        test_params_train['test_data'] = train_data_no_t
        test_params_train['post_pend'] = '_train'


        print train_params
        param_file = os.path.join(out_dir_train,'params.txt')
        all_lines = []
        for k in train_params.keys():
            str_print = '%s: %s' % (k,train_params[k])
            print str_print
            all_lines.append(str_print)
        util.writeFile(param_file,all_lines)


            
        if reconstruct:
            train_model_recon(**train_params)
            # test_model_recon(**test_params)
            # test_model_recon(**test_params_train)

        else:
            train_model(**train_params)
            test_model(**test_params)

        
    getting_accuracy.print_accuracy(out_dir_meta,pre_pend,strs_append,folds,log='log.txt')
    getting_accuracy.view_loss_curves(out_dir_meta,pre_pend,strs_append,folds,num_epochs-1)

def train_khorrami_aug_oulu(wdecay,lr,route_iter,folds=[4,9],model_name='vgg_capsule_disfa',epoch_stuff=[30,60],res=False,meta_data_dir = 'train_test_files_preprocess_vl'):
    out_dirs = []

    out_dir_meta = '../experiments/'+model_name+str(route_iter)
    num_epochs = epoch_stuff[1]
    epoch_start = 0
    dec_after = ['exp',0.96,epoch_stuff[0],1e-6]

    lr = lr
    im_resize = 110
    im_size = 96
    save_after = num_epochs
    type_data = 'three_im_no_neutral_just_strong_False'; n_classes = 6;
    criterion = 'margin'
    criterion_str = criterion

    # criterion = nn.CrossEntropyLoss()
    # criterion_str = 'crossentropy'
    
    init = False

    strs_append = '_'+'_'.join([str(val) for val in ['all_aug',criterion_str,init,'wdecay',wdecay,num_epochs]+dec_after+lr])
    pre_pend = 'oulu_96_'+meta_data_dir+'_'+type_data+'_'
    
    
    for split_num in folds:
        
        if res:
            strs_appendc = '_'.join([str(val) for val in ['all_aug','wdecay',wdecay,50,'step',50,0.1]+lr])
            out_dir_train = os.path.join(out_dir_meta,'oulu_'+type_data+'_'+str(split_num)+'_'+strs_appendc)
            model_file = os.path.join(out_dir_train,'model_49.pt')
            epoch_start = 50
        else:
            model_file = None    


        margin_params = None
        
        out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
        final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
        if os.path.exists(final_model_file):
            print 'skipping',final_model_file
            continue 

        train_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'.txt')
        test_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'test_'+str(split_num)+'.txt')
        mean_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'_mean.png')
        std_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'_std.png')
        
        mean_im = scipy.misc.imread(mean_file).astype(np.float32)
        std_im = scipy.misc.imread(std_file).astype(np.float32)
        print std_im.shape
        print np.min(std_im),np.max(std_im)
        print mean_im.shape
        print np.min(mean_im),np.max(mean_im)

        class_weights = util.get_class_weights(util.readLinesFromFile(train_file))

        # raw_input()

        list_of_to_dos = ['flip','rotate','scale_translate']
        
        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            lambda x: augmenters.augment_image(x,list_of_to_dos,mean_im,std_im,im_size),
            transforms.ToTensor(),
            lambda x: x*255.
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToTensor(),
            lambda x: x*255.
            ])

        # train_data = dataset.CK_96_Dataset_Just_Mean(train_file, mean_file, data_transforms['train'])
        # test_data = dataset.CK_96_Dataset_Just_Mean(test_file, mean_file, data_transforms['val'])
        train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
        test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
        
        network_params = dict(n_classes=n_classes,pool_type='max',r=route_iter,init=init, class_weights = class_weights)
        # if lr[0]==0:
        batch_size = 128
        batch_size_val = 128
        # else:
        #     batch_size = 32
        #     batch_size_val = 16

        util.makedirs(out_dir_train)
        
        train_params = dict(out_dir_train = out_dir_train,
                    train_data = train_data,
                    test_data = test_data,
                    batch_size = batch_size,
                    batch_size_val = batch_size_val,
                    num_epochs = num_epochs,
                    save_after = save_after,
                    disp_after = 1,
                    plot_after = 100,
                    test_after = 1,
                    lr = lr,
                    dec_after = dec_after, 
                    model_name = model_name,
                    criterion = criterion,
                    gpu_id = 0,
                    num_workers = 0,
                    model_file = model_file,
                    epoch_start = epoch_start,
                    margin_params = margin_params,
                    network_params = network_params,
                    weight_decay=wdecay)

        print train_params
        param_file = os.path.join(out_dir_train,'params.txt')
        all_lines = []
        for k in train_params.keys():
            str_print = '%s: %s' % (k,train_params[k])
            print str_print
            all_lines.append(str_print)
        util.writeFile(param_file,all_lines)

        train_model(**train_params)

        test_params = dict(out_dir_train = out_dir_train,
                model_num = num_epochs-1, 
                train_data = train_data,
                test_data = test_data,
                gpu_id = 0,
                model_name = model_name,
                batch_size_val = batch_size_val,
                criterion = criterion,
                margin_params = margin_params,
                network_params = network_params)
        test_model(**test_params)
        
    # getting_accuracy.print_accuracy(out_dir_meta,pre_pend,strs_append,folds,log='log.txt')
    getting_accuracy.view_loss_curves(out_dir_meta,pre_pend,strs_append,folds,num_epochs-1)



def main():
    
    folds = [0,1,3,4,5,6,7,8]
    
    epoch_stuff = [350,300]
    # [600,600]
    lr = [0.001,0.001,0.001]
    # res = True
    route_iter = 3

    train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3_bigclass', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = True, exp = True, non_peak = True )



    # train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = True)


    # train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3_bigrecon', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = True)

    # train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3_bigrecon', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = True)

    # train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = True)

    # oulu = True
    # # meta_data_dir = 'train_test_files_preprocess_vl'
    # meta_data_dir = 'train_test_files_preprocess_maheen_vl_gray'
    # train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = True,oulu= oulu, meta_data_dir = meta_data_dir, loss_weights = [1.,1.])

    # meta_data_dir = 'train_test_files_preprocess_maheen_vl_gray'
    # train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = True,oulu= oulu, meta_data_dir = meta_data_dir)




if __name__=='__main__':
    main()