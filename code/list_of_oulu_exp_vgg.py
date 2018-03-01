from train_test_caps import *
from torchvision import datasets, transforms
import models

import os
from helpers import util,visualize,augmenters
import random
import dataset
import numpy as np
import torch

def simple_train(wdecay,lr,route_iter,folds=[4,9],model_name='vgg_capsule_disfa',epoch_stuff=[30,60]):
    for split_num in folds:
        out_dir_meta = '../experiments/oulu_vgg_r'+str(route_iter)+'_noinit'
        num_epochs = epoch_stuff[1]
        epoch_start = 0
        dec_after = ['step',epoch_stuff[0],0.1]


        lr = lr
        im_resize = 256
        im_size = 224
        # model_name = 'vgg_capsule_disfa'
        save_after = 10
        type_data = 'three_im_no_neutral_just_strong_False'; n_classes = 6;

        # strs_append = '_'.join([str(val) for val in ['all_aug','wdecay',wdecay,pool_type,500,'step',500,0.1]+lr])
        # out_dir_train = os.path.join(out_dir_meta,'oulu_'+type_data+'_'+str(split_num)+'_'+strs_append)
        # model_file = os.path.join(out_dir_train,'model_499.pt')
        # model_file = '../experiments/oulu_vgg_r3_noinit/oulu_three_im_no_neutral_just_strong_False_4_vgg_capsule_disfa_bigprimary_all_aug_wdecay_0_100_step_100_0.1_0.0001_0.001/model_30.pt'
        # epoch_start = 30


        # type_data = 'single_im'
        model_file = None
        

        criterion = torch.nn.CrossEntropyLoss()
        margin_params = None
        

        strs_append = '_'.join([str(val) for val in [model_name,'all_aug','wdecay',wdecay,num_epochs]+dec_after+lr])
        out_dir_train = os.path.join(out_dir_meta,'oulu_'+type_data+'_'+str(split_num)+'_'+strs_append)
        print out_dir_train
        # lr[1]=lr[1]*dec_after[2]


        train_file = os.path.join('../data/Oulu_CASIA','train_test_files_preprocess_maheen_vl_color_256',type_data,'train_'+str(split_num)+'.txt')
        test_file = os.path.join('../data/Oulu_CASIA','train_test_files_preprocess_maheen_vl_color_256',type_data,'test_'+str(split_num)+'.txt')
        
        # mean_std = np.array([[129.1863,104.7624,93.5940],[1.,1.,1.]]) #rgb
        mean_std = np.array([[93.5940,104.7624,129.1863],[1.,1.,1.]]) #bgr
        print mean_std

        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((im_resize,im_resize)),
            transforms.RandomCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            lambda x: x*255,
            transforms.Normalize(mean_std[0,:],mean_std[1,:]),
            # lambda x: x[[2, 1, 0],:, : ]
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size,im_size)),
            transforms.ToTensor(),
            lambda x: x*255,
            transforms.Normalize(mean_std[0,:],mean_std[1,:]),
            # lambda x: x[[2, 1, 0],:, : ]
            ])
        data_transforms['val_center']= transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((im_resize,im_resize)),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            lambda x: x*255,
            transforms.Normalize(mean_std[0,:],mean_std[1,:]),
            # lambda x: x[[2, 1, 0],:, : ]
            ])

        train_data = dataset.Oulu_Static_Dataset(train_file, data_transforms['train'],bgr=True)
        test_data = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val'],bgr=True)
        test_data_center = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val_center'],bgr=True)
        
        network_params = dict(n_classes=n_classes,r=route_iter,init=False)
        if lr[0]==0:
            batch_size = 128
            batch_size_val = 128
        else:
            batch_size = 32
            batch_size_val = 16

        util.makedirs(out_dir_train)
        
        train_params = dict(out_dir_train = out_dir_train,
                    train_data = train_data,
                    test_data = test_data,
                    batch_size = batch_size,
                    batch_size_val = batch_size_val,
                    num_epochs = num_epochs,
                    save_after = save_after,
                    disp_after = 1,
                    plot_after = 10,
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
        
        test_params['test_data'] = test_data_center
        test_params['post_pend'] = '_center'
        test_model(**test_params)

def simple_train_preprocessed(wdecay,lr,route_iter,folds=[4,9],model_name='vgg_capsule_disfa',epoch_stuff=[30,60],res=False):
    for split_num in folds:
        out_dir_meta = '../experiments/oulu_vgg_r'+str(route_iter)+'_noinit_preprocessed'
        num_epochs = epoch_stuff[1]
        epoch_start = 0
        dec_after = ['step',epoch_stuff[0],0.1]

        # data/Oulu_CASIA/train_test_files_preprocess_vl/
        lr = lr
        im_resize = 256
        im_size = 224
        # model_name = 'vgg_capsule_disfa'
        save_after = 10
        type_data = 'three_im_no_neutral_just_strong_False'; n_classes = 6;

        if res:
            strs_append = '_'.join([str(val) for val in [model_name,'all_aug','wdecay',wdecay,50,'step',50,0.1]+lr])
            out_dir_train = os.path.join(out_dir_meta,'oulu_'+type_data+'_'+str(split_num)+'_'+strs_append)
            model_file = os.path.join(out_dir_train,'model_49.pt')
            epoch_start = 50
        else:
            model_file = None    

        # model_file = '../experiments/oulu_vgg_r3_noinit/oulu_three_im_no_neutral_just_strong_False_4_vgg_capsule_disfa_bigprimary_all_aug_wdecay_0_100_step_100_0.1_0.0001_0.001/model_30.pt'
        # epoch_start = 30


        # type_data = 'single_im'
        
        

        criterion = torch.nn.CrossEntropyLoss()
        margin_params = None
        

        strs_append = '_'.join([str(val) for val in [model_name,'all_aug','wdecay',wdecay,num_epochs]+dec_after+lr])
        out_dir_train = os.path.join(out_dir_meta,'oulu_'+type_data+'_'+str(split_num)+'_'+strs_append)
        print out_dir_train
        final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
        if os.path.exists(final_model_file):
            print 'skipping',final_model_file
            continue 
        # lr[1]=lr[1]*dec_after[2]


        train_file = os.path.join('../data/Oulu_CASIA','train_test_files_preprocess_vl',type_data,'train_'+str(split_num)+'.txt')
        test_file = os.path.join('../data/Oulu_CASIA','train_test_files_preprocess_vl',type_data,'test_'+str(split_num)+'.txt')
        
        # mean_std = np.array([[129.1863,104.7624,93.5940],[1.,1.,1.]]) #rgb
        mean_std = np.array([[128.,128.,128.],[1.,1.,1.]]) #bgr
        print mean_std

        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_resize,im_resize)),
            transforms.RandomCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            lambda x: x*255,
            transforms.Normalize(mean_std[0,:],mean_std[1,:]),
            # lambda x: x[[2, 1, 0],:, : ]
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size,im_size)),
            transforms.ToTensor(),
            lambda x: x*255,
            transforms.Normalize(mean_std[0,:],mean_std[1,:]),
            # lambda x: x[[2, 1, 0],:, : ]
            ])
        data_transforms['val_center']= transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((im_resize,im_resize)),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            lambda x: x*255,
            transforms.Normalize(mean_std[0,:],mean_std[1,:]),
            # lambda x: x[[2, 1, 0],:, : ]
            ])

        train_data = dataset.Oulu_Static_Dataset(train_file, data_transforms['train'],color=True)
        test_data = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val'],color=True)
        test_data_center = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val_center'],color=True)
        
        network_params = dict(n_classes=n_classes,r=route_iter,init=False)
        if lr[0]==0:
            batch_size = 128
            batch_size_val = 128
        else:
            batch_size = 32
            batch_size_val = 16

        util.makedirs(out_dir_train)
        
        train_params = dict(out_dir_train = out_dir_train,
                    train_data = train_data,
                    test_data = test_data,
                    batch_size = batch_size,
                    batch_size_val = batch_size_val,
                    num_epochs = num_epochs,
                    save_after = save_after,
                    disp_after = 1,
                    plot_after = 10,
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
        
        test_params['test_data'] = test_data_center
        test_params['post_pend'] = '_center'
        test_model(**test_params)



def main():
    
    folds = range(10)
    epoch_stuff = [100,100]
    lr = [0.00001,0.0001]
    res=True
    simple_train_preprocessed(0, lr, 1, folds= folds, model_name='vgg_capsule_disfa_bigprimary', epoch_stuff=epoch_stuff,res=res)
    simple_train_preprocessed(0, lr, 1, folds= folds, model_name='vgg_capsule_disfa_bigclass', epoch_stuff=epoch_stuff,res=res)
    simple_train_preprocessed(0, lr, 1, folds= folds, model_name='vgg_capsule_disfa', epoch_stuff=epoch_stuff,res=res)

    folds = range(10)
    epoch_stuff = [100,100]
    lr = [0.0001,0.0001]
    res=False
    simple_train_preprocessed(0, lr, 1, folds= folds, model_name='vgg_capsule_disfa_bigprimary', epoch_stuff=epoch_stuff,res=res)
    simple_train_preprocessed(0, lr, 1, folds= folds, model_name='vgg_capsule_disfa_bigclass', epoch_stuff=epoch_stuff,res=res)
    simple_train_preprocessed(0, lr, 1, folds= folds, model_name='vgg_capsule_disfa', epoch_stuff=epoch_stuff,res=res)    
    # simple_train(0,[0.0001,0.001],3,[4],'vgg_capsule_disfa_bigprimary',[15,30])
    # simple_train(0,[0.0001,0.001],3,[4],'vgg_capsule_disfa_bigclass',[15,30])
    # simple_train(0,[0.0001,0.001],1,[9],'vgg_capsule_disfa_bigprimary',[15,30])
    # simple_train(0,[0.0001,0.001],1,[9],'vgg_capsule_disfa_bigclass',[15,30])
    
    # simple_train(0,[0.0001,0.001],2,[4],'vgg_capsule_disfa_bigprimary',[15,30])
    # simple_train(0,[0.0001,0.001],2,[4],'vgg_capsule_disfa_bigclass',[15,30])

    # simple_train(0,[0.0001,0.001],1,[4],'vgg_capsule_disfa_bigprimary',[15,30])
    # simple_train(0,[0.0001,0.001],1,[4],'vgg_capsule_disfa_bigclass',[15,30])
    


if __name__=='__main__':
    main()