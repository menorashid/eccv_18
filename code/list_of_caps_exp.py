from train_test_caps import *
from torchvision import datasets, transforms
import models

import os
from helpers import util,visualize,augmenters
import random
import dataset
import numpy as np

def bigger_exp():
    out_dir_meta = '../experiments/dynamic_capsules/'
    num_epochs = 100
    dec_after = ['exp',0.96,100,1e-6]
    lr = 0.001
    split_num = 0
    im_size = 28
    # margin_params = {'step':1,'start':0.2}

    strs_append = '_'.join([str(val) for val in [num_epochs,dec_after[0],lr]])
    
    out_dir_train = os.path.join(out_dir_meta,'ck_bigger_'+str(split_num)+'_'+strs_append)
    print out_dir_train

    train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
    test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
    mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
    std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'

    data_transforms = {}
    data_transforms['train']= transforms.Compose([
        # lambda x: augmenters.random_crop(x,32),
        lambda x: augmenters.horizontal_flip(x),
        transforms.ToTensor(),
        lambda x: x*255.
    ])
    data_transforms['val']= transforms.Compose([
        # lambda x: augmenters.crop_center(x,32,32),
        transforms.ToTensor(),
        lambda x: x*255.
        ])

    im_size = 48
    train_data = dataset.CK_RS_Dataset(train_file, mean_file, std_file, im_size, data_transforms['train'])
    test_data = dataset.CK_RS_Dataset(test_file, mean_file, std_file, im_size, data_transforms['val'])
    # train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
    # test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    
    network_params = dict(n_classes=8,
                        conv_layers = 
                        # None,
                        [[256,5,2]],
                        caps_layers = 
                        # None,
                        [[32,8,5,2],[32,8,3,2],[32,8,4,1]],
                        r=3)
    
    batch_size = 64
    batch_size_val = 32


    util.makedirs(out_dir_train)
    
    train_params = dict(out_dir_train = out_dir_train,
                train_data = train_data,
                test_data = test_data,
                batch_size = batch_size,
                batch_size_val = batch_size_val,
                num_epochs = num_epochs,
                save_after = 1,
                disp_after = 1,
                plot_after = 10,
                test_after = 1,
                lr = lr,
                dec_after = dec_after, 
                model_name = 'dynamic_capsules',
                criterion = 'margin',
                gpu_id = 1,
                num_workers = 0,
                model_file = None,
                epoch_start = 0,
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


def baseline_exp():
    out_dir_meta = '../experiments/dynamic_capsules/'
    num_epochs = 108
    dec_after = ['exp',0.96,50,1e-6]
    lr = 0.001
    split_num = 0
    im_size = 28
    # margin_params = {'step':1,'start':0.2}

    strs_append = '_'.join([str(val) for val in [num_epochs,dec_after[0],lr]])
    
    out_dir_train = os.path.join(out_dir_meta,'baseline_conv_ck_'+str(split_num)+'_'+strs_append)
    print out_dir_train

    train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
    test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
    mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
    std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'

    data_transforms = {}
    data_transforms['train']= transforms.Compose([
        # lambda x: augmenters.random_crop(x,32),
        lambda x: augmenters.horizontal_flip(x),
        transforms.ToTensor(),
        lambda x: x*255.
    ])
    data_transforms['val']= transforms.Compose([
        # lambda x: augmenters.crop_center(x,32,32),
        transforms.ToTensor(),
        lambda x: x*255.
        ])

    train_data = dataset.CK_RS_Dataset(train_file, mean_file, std_file, im_size, data_transforms['train'])
    test_data = dataset.CK_RS_Dataset(test_file, mean_file, std_file, im_size, data_transforms['val'])
    # train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
    # test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    
    network_params = dict(n_classes=8)
    
    batch_size = 128
    batch_size_val = 128


    util.makedirs(out_dir_train)
    
    train_params = dict(out_dir_train = out_dir_train,
                train_data = train_data,
                test_data = test_data,
                batch_size = batch_size,
                batch_size_val = batch_size_val,
                num_epochs = num_epochs,
                save_after = 10,
                disp_after = 1,
                plot_after = 10,
                test_after = 1,
                lr = lr,
                dec_after = dec_after, 
                model_name = 'convolution_baseline_tf',
                gpu_id = 2,
                num_workers = 0,
                model_file = None,
                epoch_start = 0,
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

    # train_data = dataset.CK_RS_Dataset(train_file, mean_file, std_file, im_size, data_transforms['val'])
    # save_output_capsules(out_dir_train,
    #             num_epochs-1,
    #             train_data,
    #             test_data,
    #             model_name = 'dynamic_capsules',
    #             batch_size_val =batch_size_val,
    #             network_params = network_params)
    

def main():
    # baseline_exp()

    # return
    out_dir_meta = '../experiments/dynamic_capsules_fixed_recon/'
    num_epochs = 108
    dec_after = ['exp',0.96,50,1e-6]
    lr = 0.001
    split_num = 0
    im_size = 28
    save_after = 50
    reconstruct = True
    # margin_params = {'step':1,'start':0.2}

    strs_append = '_'.join([str(val) for val in [reconstruct,num_epochs]+dec_after+[lr]])
    
    out_dir_train = os.path.join(out_dir_meta,'ck_'+str(split_num)+'_last32_'+strs_append)
    print out_dir_train

    train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
    test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
    mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
    std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'

    data_transforms = {}
    data_transforms['train']= transforms.Compose([
        # lambda x: augmenters.random_crop(x,32),
        lambda x: augmenters.horizontal_flip(x),
        transforms.ToTensor(),
        lambda x: x*255.
    ])
    data_transforms['val']= transforms.Compose([
        # lambda x: augmenters.crop_center(x,32,32),
        transforms.ToTensor(),
        lambda x: x*255.
        ])

    train_data = dataset.CK_RS_Dataset(train_file, mean_file, std_file, im_size, data_transforms['train'])
    test_data = dataset.CK_RS_Dataset(test_file, mean_file, std_file, im_size, data_transforms['val'])
    # train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
    # test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    
    network_params = dict(n_classes=8,
                        conv_layers = 
                        None,
                        # [[256,5,2]],
                        caps_layers = 
                        # None,
                        [[32,8,9,2],[8,32,6,1]],
                        r=3,
                        reconstruct=reconstruct)
    
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
                model_name = 'dynamic_capsules',
                criterion = 'margin',
                gpu_id = 1,
                num_workers = 0,
                model_file = None,
                epoch_start = 0,
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

    train_data = dataset.CK_RS_Dataset(train_file, mean_file, std_file, im_size, data_transforms['val'])
    save_output_capsules(out_dir_train,
                num_epochs-1,
                train_data,
                test_data,
                model_name = 'dynamic_capsules',
                batch_size_val =batch_size_val,
                network_params = network_params)

    save_perturbed_images(out_dir_train,
                num_epochs - 1,
                train_data,
                test_data,
                model_name = 'dynamic_capsules',
                batch_size_val =batch_size_val,
                network_params = network_params)

if __name__=='__main__':
    main()