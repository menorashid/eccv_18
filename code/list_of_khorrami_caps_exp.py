from train_test_caps import *
from torchvision import datasets, transforms
import models

import os
from helpers import util,visualize,augmenters
import random
import dataset
import numpy as np


def khorrami_exp():
    for split_num in range(0,1):
        out_dir_meta = '../experiments/khorrami_caps_k7_s3_smaller/'
        num_epochs = 50
        epoch_start = 0
        dec_after = ['exp',0.96,25,1e-6]
        # dec_after = ['step',50,0.5]
        lr = [0.001]
        pool_type = 'max'
        im_size = 96
        model_name = 'khorrami_capsule'
        save_after = 50
        model_file=None    

        strs_append = '_'.join([str(val) for val in ['justflip',pool_type,num_epochs]+dec_after+lr])
        out_dir_train = os.path.join(out_dir_meta,'ck_'+str(split_num)+'_'+strs_append)
        print out_dir_train


        train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
        test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
        mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
        std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'
        
        mean_im = scipy.misc.imread(mean_file).astype(np.float32)
        std_im = scipy.misc.imread(std_file).astype(np.float32)
        std_im[std_im==0]=1.
        list_of_to_dos = ['flip']
        # ,'rotate','scale_translate']'pixel_augment',
        # 'flip','rotate','scale_translate']

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

        train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
        test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
        
        network_params = dict(n_classes=8,pool_type=pool_type,r=3,init=False)
        
        batch_size = 128
        batch_size_val = 128


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
                    criterion = 'margin',
                    gpu_id = 2,
                    num_workers = 0,
                    model_file = model_file,
                    epoch_start = epoch_start,
                    network_params = network_params)

        print train_params
        param_file = os.path.join(out_dir_train,'params.txt')
        all_lines = []
        for k in train_params.keys():
            str_print = '%s: %s' % (k,train_params[k])
            print str_print
            all_lines.append(str_print)
        util.writeFile(param_file,all_lines)

        train_model(**train_params)

def khorrami_full_exp():
    for split_num in range(0,1):
        out_dir_meta = '../experiments/khorrami_full_capsule/'
        num_epochs = 100
        epoch_start = 0
        dec_after = ['exp',0.96,200,1e-6]
        # dec_after = ['step',50,0.5]
        lr = [0.001]
        pool_type = 'max'
        im_size = 96
        model_name = 'khorrami_full_capsule'
        save_after = 50
        model_file=None    

        strs_append = '_'.join([str(val) for val in ['justflip',pool_type,num_epochs]+dec_after+lr])
        out_dir_train = os.path.join(out_dir_meta,'ck_'+str(split_num)+'_'+strs_append)
        print out_dir_train


        train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
        test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
        mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
        std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'
        
        mean_im = scipy.misc.imread(mean_file).astype(np.float32)
        std_im = scipy.misc.imread(std_file).astype(np.float32)
        std_im[std_im==0]=1.
        list_of_to_dos = ['flip']
        # ,'rotate','scale_translate','pixel_augment']
        # 'flip','rotate','scale_translate']

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

        train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
        test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
        
        network_params = dict(n_classes=8, conv_layers = [[64,5,2]],caps_layers=[[16,8,5,2],[32,8,7,3],[8,16,5,1]], r=3, init=False)
        
        batch_size = 32
        batch_size_val = 4


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
                    test_after = num_epochs-1,
                    lr = lr,
                    dec_after = dec_after, 
                    model_name = model_name,
                    criterion = 'margin',
                    gpu_id = 2,
                    num_workers = 0,
                    model_file = model_file,
                    epoch_start = epoch_start,
                    network_params = network_params)

        print train_params
        param_file = os.path.join(out_dir_train,'params.txt')
        all_lines = []
        for k in train_params.keys():
            str_print = '%s: %s' % (k,train_params[k])
            print str_print
            all_lines.append(str_print)
        util.writeFile(param_file,all_lines)

        train_model(**train_params)

def khorrami_search_lr():
    split_num = 0
    out_dir_meta = '../experiments/khorrami_caps_k7_s3_smaller/'
    num_epochs = 300
    epoch_start = 0
    lr = [0.001]
    dec_after = ['reduce','max',0.96,5,1e-6]
    # ['exp',0.96,25,1e-6]
    # dec_after = ['step',50,0.5]
    
    pool_type = 'max'
    im_size = 96
    model_name = 'khorrami_capsule'
    save_after = 50
    model_file=None    

    strs_append = '_'.join([str(val) for val in ['flippixel',pool_type,num_epochs]+dec_after+lr])
    out_dir_train = os.path.join(out_dir_meta,'ck_'+str(split_num)+'_'+strs_append)
    print out_dir_train


    train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
    test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
    mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
    std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'
    
    mean_im = scipy.misc.imread(mean_file).astype(np.float32)
    std_im = scipy.misc.imread(std_file).astype(np.float32)
    std_im[std_im==0]=1.
    list_of_to_dos = ['flip','pixel_augment']
    # 'scale_translate']
    # ,'rotate','scale_translate',
    # 'flip','rotate','scale_translate']

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

    train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
    test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    
    network_params = dict(n_classes=8,pool_type=pool_type,r=3,init=False)
    
    batch_size = 128
    batch_size_val = 128


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
                criterion = 'margin',
                gpu_id = 2,
                num_workers = 0,
                model_file = model_file,
                epoch_start = epoch_start,
                network_params = network_params)

    print train_params
    param_file = os.path.join(out_dir_train,'params.txt')
    all_lines = []
    for k in train_params.keys():
        str_print = '%s: %s' % (k,train_params[k])
        print str_print
        all_lines.append(str_print)
    util.writeFile(param_file,all_lines)

    train_model(**train_params)

def caps_for_96():
    split_num = 0
    out_dir_meta = '../experiments/khorrami_caps_for_96/'
    num_epochs = 200
    epoch_start = 0
    lr = [0.001]
    dec_after = ['reduce','max',0.96,5,1e-6]
    # ['exp',0.96,25,1e-6]
    # 
    # 
    # dec_after = ['step',50,0.5]
    
    pool_type = 'max'
    im_size = 96
    model_name = 'caps_for_96'
    save_after = 50
    model_file=None    

    strs_append = '_'.join([str(val) for val in ['flipscaletranslate',pool_type,num_epochs]+dec_after+lr])
    out_dir_train = os.path.join(out_dir_meta,'ck_'+str(split_num)+'_'+strs_append)
    print out_dir_train


    train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
    test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
    mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
    std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'
    
    mean_im = scipy.misc.imread(mean_file).astype(np.float32)
    std_im = scipy.misc.imread(std_file).astype(np.float32)
    std_im[std_im==0]=1.
    list_of_to_dos = ['flip','scale_translate']
    # ,'pixel_augment']
    # ,'rotate','scale_translate',
    # 'flip','rotate','scale_translate']

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

    train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
    test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    
    network_params = dict(n_classes=8,
                        conv_layers = [[256,11,5,5]],
                        caps_layers = [[32,8,9,2],[8,16,6,1]], r=3, init=False)
    
    batch_size = 128
    batch_size_val = 128

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
                criterion = 'margin',
                gpu_id = 2,
                num_workers = 0,
                model_file = model_file,
                epoch_start = epoch_start,
                network_params = network_params)

    print train_params
    param_file = os.path.join(out_dir_train,'params.txt')
    all_lines = []
    for k in train_params.keys():
        str_print = '%s: %s' % (k,train_params[k])
        print str_print
        all_lines.append(str_print)
    util.writeFile(param_file,all_lines)

    train_model(**train_params)


def khorrami_new_aug_method():
    for split_num in range(0,1):
        out_dir_meta = '../experiments/khorrami_caps_k7_s3_new_aug/'
        num_epochs = 1500
        # epoch_start = 0
        # dec_after = ['exp',0.96,25,1e-6]
        dec_after = ['reduce','max',0.96,20,1e-6]
        # dec_after = ['step',50,0.5]
        lr = [0.000542]
        pool_type = 'max'
        im_size = 96
        model_name = 'khorrami_capsule'
        save_after = 50
        model_file='../experiments/khorrami_caps_k7_s3_new_aug/ck_0_max_500_reduce_max_0.96_20_1e-06_0.001/model_499.pt'
        epoch_start = 500

        strs_append = '_'.join([str(val) for val in [pool_type,num_epochs]+dec_after+lr])
        out_dir_train = os.path.join(out_dir_meta,'ck_'+str(split_num)+'_'+strs_append)
        print out_dir_train


        train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
        test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
        mean_std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean_std_val_0_1.npy'

        mean_std = np.load(mean_std_file)

        print mean_std

        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(102),
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
            ])

        train_data = dataset.CK_96_New_Dataset(train_file, data_transforms['train'])
        test_data = dataset.CK_96_New_Dataset(test_file, data_transforms['val'])
            
        network_params = dict(n_classes=8,pool_type=pool_type,r=3,init=False)
        
        batch_size = 128
        batch_size_val = 128


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
                    criterion = 'margin',
                    gpu_id = 2,
                    num_workers = 0,
                    model_file = model_file,
                    epoch_start = epoch_start,
                    network_params = network_params)

        print train_params
        param_file = os.path.join(out_dir_train,'params.txt')
        all_lines = []
        for k in train_params.keys():
            str_print = '%s: %s' % (k,train_params[k])
            print str_print
            all_lines.append(str_print)
        util.writeFile(param_file,all_lines)

        train_model(**train_params)

def explore_new_architecture():
    for split_num in range(0,1):
        out_dir_meta = '../experiments/caps_heavy_48/'
        num_epochs = 1000
        # epoch_start = 0
        dec_after = ['exp',0.96,100,1e-6]
        # dec_after = ['reduce','max',0.96,5,1e-6]
        # dec_after = ['step',50,0.5]
        lr = [0.001]
        pool_type = 'nopool'
        im_size = 96
        model_name = 'caps_for_96'
        save_after = 100
        model_file = None
        # '../experiments/khorrami_caps_k7_s3_new_aug/ck_0_max_500_reduce_max_0.96_20_1e-06_0.001/model_499.pt'
        epoch_start = 0

        strs_append = '_'.join([str(val) for val in ['justflip',pool_type,num_epochs]+dec_after+lr])
        out_dir_train = os.path.join(out_dir_meta,'ck_'+str(split_num)+'_'+strs_append)
        print out_dir_train


        train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
        test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
        mean_std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean_std_val_0_1.npy'

        mean_std = np.load(mean_std_file)

        print mean_std

        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(50),
            transforms.RandomResizedCrop(48),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(48),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
            ])

        train_data = dataset.CK_96_New_Dataset(train_file, data_transforms['train'])
        test_data = dataset.CK_96_New_Dataset(test_file, data_transforms['val'])
            
        network_params = dict(n_classes=8,
            pool_type=pool_type,
            conv_layers = [[256,7,3,0]],
            # [[32,5,2,0],[64,5,2,0]],
            caps_layers = [[32,8,7,3],[8,32,3,1]],
            # [[32,8,5,2],[8,16,5,1]],
            r=3, init=False)
        
        batch_size = 128
        batch_size_val = 128


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
                    criterion = 'margin',
                    gpu_id = 1,
                    num_workers = 0,
                    model_file = model_file,
                    epoch_start = epoch_start,
                    network_params = network_params)

        print train_params
        param_file = os.path.join(out_dir_train,'params.txt')
        all_lines = []
        for k in train_params.keys():
            str_print = '%s: %s' % (k,train_params[k])
            print str_print
            all_lines.append(str_print)
        util.writeFile(param_file,all_lines)

        train_model(**train_params)


def main():
    explore_new_architecture()
    # khorrami_new_aug_method()

if __name__=='__main__':
    main()