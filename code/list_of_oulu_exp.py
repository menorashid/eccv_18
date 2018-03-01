from train_test_caps import *
from torchvision import datasets, transforms
import models

import os
from helpers import util,visualize,augmenters
import random
import dataset
import numpy as np


def increasing_spread_improving_r1(wdecay,lr,margin):
    for split_num in [9]:
        out_dir_meta = '../experiments/oulu_improving_by_spread_'+str(margin)+'_r1/'
        route_iter = 1
        num_epochs = 1000
        epoch_start = 0
        # dec_after = ['exp',0.96,3,1e-6]
        dec_after = ['step',1000,0.1]


        lr = lr
        # [0.001]
        pool_type = 'max'
        im_size = 96
        model_name = 'khorrami_capsule'
        save_after = 50
        type_data = 'three_im_no_neutral_just_strong'; n_classes = 6;

        # strs_append = '_'.join([str(val) for val in ['all_aug','wdecay',wdecay,pool_type,500,'step',500,0.1]+lr])
        # out_dir_train = os.path.join(out_dir_meta,'oulu_'+type_data+'_'+str(split_num)+'_'+strs_append)
        # model_file = os.path.join(out_dir_train,'model_499.pt')
        # type_data = 'single_im'
        model_file = None
        

        criterion = 'spread'
        margin_params = dict(end_epoch=int(num_epochs*margin),decay_steps=5,init_margin = margin, max_margin = margin)
        

        strs_append = '_'.join([str(val) for val in ['all_aug','wdecay',wdecay,pool_type,num_epochs]+dec_after+lr])
        out_dir_train = os.path.join(out_dir_meta,'oulu_'+type_data+'_'+str(split_num)+'_'+strs_append)
        print out_dir_train

        train_file = os.path.join('../data/Oulu_CASIA','train_test_files_preprocess_maheen_vl_gray',type_data,'train_'+str(split_num)+'.txt')
        test_file = os.path.join('../data/Oulu_CASIA','train_test_files_preprocess_maheen_vl_gray',type_data,'test_'+str(split_num)+'.txt')
        mean_std_file = os.path.join('../data/Oulu_CASIA','train_test_files_preprocess_maheen_vl_gray',type_data,'train_'+str(split_num)+'_mean_std_val_0_1.npy')
        
        class_weights = util.get_class_weights(util.readLinesFromFile(train_file))

        mean_std = np.load(mean_std_file)

        print mean_std

        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((102,102)),
            transforms.RandomCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size,im_size)),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
            ])
        data_transforms['val_center']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((102,102)),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
            ])

        train_data = dataset.Oulu_Static_Dataset(train_file, data_transforms['train'])
        test_data = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val'])
        test_data_center = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val_center'])
        
        network_params = dict(n_classes=n_classes,pool_type=pool_type,r=route_iter,init=True,class_weights = class_weights)
        
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


def khorrami_with_val():
    for split_num in range(0,10):
    # range(0,10):
        out_dir_meta = '../experiments/oulu_r1_hopeful/'
        route_iter = 1
        num_epochs = 500
        epoch_start = 0
        # dec_after = ['exp',0.96,3,1e-6]
        # dec_after = ['step',500,0.1]
        # wdecay_all_aug_max_500_reduce_min_0.1_50_0.0001_0.001
        dec_after = ['reduce','min',0.1,50,1e-4]
        # dec_after = ['reduce','max',0.96,5,1e-6]
        # 'reduce':
        #     exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=dec_after[1], factor=dec_after[2], patience=dec_after[3],min_lr=dec_after[4])


        lr = [0.001]
        pool_type = 'max'
        im_size = 96
        model_name = 'khorrami_capsule'
        save_after = 50
        model_file = None    
        # type_data = 'single_im'
        type_data = 'three_im_no_neutral_just_strong_True'; n_classes = 6;
        train_test_folder = 'train_test_files_preprocess_maheen_vl_gray'

        criterion = 'spread'
        margin_params = dict(end_epoch=int(num_epochs*0.9),decay_steps=5,max_margin = 0.2)
        # None
        # criterion = 'margin'

        strs_append = '_'.join([str(val) for val in ['wdecay','all_aug',pool_type,num_epochs]+dec_after+lr])
        out_dir_train = os.path.join(out_dir_meta,'oulu_'+type_data+'_'+str(split_num)+'_'+strs_append)
        print out_dir_train

        train_file = os.path.join('../data/Oulu_CASIA',train_test_folder,type_data,'train_'+str(split_num)+'.txt')
        val_file = os.path.join('../data/Oulu_CASIA',train_test_folder,type_data,'val_'+str(split_num)+'.txt')
        test_file = os.path.join('../data/Oulu_CASIA',train_test_folder,type_data,'test_'+str(split_num)+'.txt')
        mean_std_file = os.path.join('../data/Oulu_CASIA',train_test_folder,type_data,'train_'+str(split_num)+'_mean_std_val_0_1.npy')
        
        class_weights = util.get_class_weights(util.readLinesFromFile(train_file))

        mean_std = np.load(mean_std_file)

        print mean_std

        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((110,110)),
            transforms.RandomCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size,im_size)),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
            ])
        data_transforms['val_center']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((110,110)),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
            ])

        train_data = dataset.Oulu_Static_Dataset(train_file, data_transforms['train'])
        val_data = dataset.Oulu_Static_Dataset(val_file, data_transforms['val'])
        test_data = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val'])
        test_data_center = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val_center'])
        
        network_params = dict(n_classes=n_classes,pool_type=pool_type,r=route_iter,init=False,class_weights = class_weights)
        
        batch_size = 128
        batch_size_val = 128


        util.makedirs(out_dir_train)
        
        train_params = dict(out_dir_train = out_dir_train,
                    train_data = train_data,
                    test_data = val_data,
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
                    network_params = network_params)

        print train_params
        param_file = os.path.join(out_dir_train,'params.txt')
        all_lines = []
        for k in train_params.keys():
            str_print = '%s: %s' % (k,train_params[k])
            print str_print
            all_lines.append(str_print)
        util.writeFile(param_file,all_lines)

        # train_model(**train_params)

        test_params = dict(out_dir_train = out_dir_train,
                model_num = num_epochs - 1, 
                train_data = train_data,
                test_data = test_data,
                gpu_id = 0,
                model_name = model_name,
                batch_size_val = batch_size_val,
                criterion = criterion,
                margin_params = margin_params,
                network_params = network_params)
        test_model(**test_params)
        
        model_nums = range(save_after,num_epochs,save_after)
        model_nums.append(num_epochs-1)
        print model_nums
        test_params['model_nums']=model_nums
        test_model_list_models(**test_params)

        test_params['test_data'] = test_data_center
        test_params['post_pend'] = '_center'
        test_model(**test_params)
        test_model_list_models(**test_params)

        if dec_after[0]=='reduce':
            test_params = dict(out_dir_train = out_dir_train,
                    model_num = 'bestVal', 
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
        

def test_center_crop():
    
    split_num = 0
    out_dir_meta = '../experiments/khorrami_caps_k7_s3_oulu_spread_0.2/'
    num_epochs = 300
    epoch_start = 0
    # dec_after = ['exp',0.96,5,1e-6]
    dec_after = ['step',150,0.1]

    lr = [0.001]
    pool_type = 'nopool'
    im_size = 96
    model_name = 'khorrami_capsule'
    save_after = num_epochs
    model_file=None    
    # type_data = 'single_im'
    type_data = 'single_im'
    criterion = 'spread'
    margin_params = {'start':0.2}
    strs_append = '_'.join([str(val) for val in ['all_aug',pool_type,num_epochs]+dec_after+lr])
    out_dir_train = os.path.join(out_dir_meta,'oulu_'+type_data+'_'+str(split_num)+'_'+strs_append)
    print out_dir_train

    train_file = os.path.join('../data/Oulu_CASIA','train_test_files',type_data,'train_'+str(split_num)+'.txt')
    test_file = os.path.join('../data/Oulu_CASIA','train_test_files',type_data,'test_'+str(split_num)+'.txt')
    mean_std_file = os.path.join('../data/Oulu_CASIA','train_test_files',type_data,'train_'+str(split_num)+'_mean_std_val_0_1.npy')
    
    class_weights = util.get_class_weights(util.readLinesFromFile(train_file))

    mean_std = np.load(mean_std_file)

    data_transforms = {}
    data_transforms['train']= transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((102,102)),
        transforms.RandomCrop(im_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
    ])
    data_transforms['val']= transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((102,102)),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
        ])


    train_data = dataset.Oulu_Static_Dataset(train_file, data_transforms['train'])
    test_data = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val'])
    
    network_params = dict(n_classes=7,pool_type=pool_type,r=3,init=False,class_weights = class_weights)
    
    batch_size = 128
    batch_size_val = 128


    util.makedirs(out_dir_train)
    
    test_params = dict(out_dir_train = out_dir_train,
            model_num = num_epochs-1, 
            train_data = train_data,
            test_data = test_data,
            gpu_id = 0,
            model_name = model_name,
            batch_size_val = batch_size_val,
            criterion = criterion,
            margin_params = margin_params,
            network_params = network_params,
            post_pend='center')
    test_model(**test_params)

def khorrami_exp_spread():
    for split_num in range(0,10):
        out_dir_meta = '../experiments/khorrami_caps_k7_s3_oulu_spread_0.2_vl_gray_r_3_init_correct_out/'
        route_iter = 3
        num_epochs = 300
        epoch_start = 0
        # dec_after = ['exp',0.96,3,1e-6]
        dec_after = ['step',300,0.1]


        lr = [0.001]
        pool_type = 'max'
        im_size = 96
        model_name = 'khorrami_capsule'
        save_after = num_epochs
        model_file = None    
        # type_data = 'single_im'
        type_data = 'three_im_no_neutral_just_strong'; n_classes = 6;

        criterion = 'spread'
        margin_params = dict(end_epoch=int(num_epochs*0.9),decay_steps=5,max_margin = 0.2)
        

        strs_append = '_'.join([str(val) for val in ['all_aug',pool_type,num_epochs]+dec_after+lr])
        out_dir_train = os.path.join(out_dir_meta,'oulu_'+type_data+'_'+str(split_num)+'_'+strs_append)
        print out_dir_train

        train_file = os.path.join('../data/Oulu_CASIA','train_test_files_preprocess_maheen_vl_gray',type_data,'train_'+str(split_num)+'.txt')
        test_file = os.path.join('../data/Oulu_CASIA','train_test_files_preprocess_maheen_vl_gray',type_data,'test_'+str(split_num)+'.txt')
        mean_std_file = os.path.join('../data/Oulu_CASIA','train_test_files_preprocess_maheen_vl_gray',type_data,'train_'+str(split_num)+'_mean_std_val_0_1.npy')
        
        class_weights = util.get_class_weights(util.readLinesFromFile(train_file))

        mean_std = np.load(mean_std_file)

        print mean_std

        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((102,102)),
            transforms.RandomCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size,im_size)),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
            ])
        data_transforms['val_center']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((102,102)),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
            ])

        train_data = dataset.Oulu_Static_Dataset(train_file, data_transforms['train'])
        test_data = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val'])
        test_data_center = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val_center'])
        
        network_params = dict(n_classes=n_classes,pool_type=pool_type,r=route_iter,init=True,class_weights = class_weights)
        
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
                    criterion = criterion,
                    gpu_id = 0,
                    num_workers = 0,
                    model_file = model_file,
                    epoch_start = epoch_start,
                    margin_params = margin_params,
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

def khorrami_exp():
    for split_num in range(0,1):
        out_dir_meta = '../experiments/khorrami_caps_k7_s3_oulu_class_weights_no_norm/'
        num_epochs = 300
        epoch_start = 0
        # dec_after = ['exp',0.96,5,1e-6]
        dec_after = ['step',150,0.1]

        lr = [0.001]
        pool_type = 'nopool'
        im_size = 96
        model_name = 'khorrami_capsule'
        save_after = num_epochs
        model_file=None    
        type_data = 'single_im'

        strs_append = '_'.join([str(val) for val in ['all_aug',pool_type,num_epochs]+dec_after+lr])
        out_dir_train = os.path.join(out_dir_meta,'oulu_'+type_data+'_'+str(split_num)+'_'+strs_append)
        print out_dir_train

        train_file = os.path.join('../data/Oulu_CASIA','train_test_files',type_data,'train_'+str(split_num)+'.txt')
        test_file = os.path.join('../data/Oulu_CASIA','train_test_files',type_data,'test_'+str(split_num)+'.txt')
        mean_std_file = os.path.join('../data/Oulu_CASIA','train_test_files',type_data,'train_'+str(split_num)+'_mean_std_val_0_1.npy')
        
        class_weights = util.get_class_weights(util.readLinesFromFile(train_file))

        mean_std = np.load(mean_std_file)

        print mean_std

        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((102,102)),
            transforms.RandomCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size,im_size)),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
            ])


        train_data = dataset.Oulu_Static_Dataset(train_file, data_transforms['train'])
        test_data = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val'])
        
        network_params = dict(n_classes=7,pool_type=pool_type,r=3,init=False,class_weights = class_weights)
        
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
                    gpu_id = 0,
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

        test_params = dict(out_dir_train = out_dir_train,
                model_num = num_epochs-1, 
                train_data = train_data,
                test_data = test_data,
                gpu_id = 0,
                model_name = model_name,
                batch_size_val = batch_size_val,
                criterion = 'margin',
                network_params = network_params)
        test_model(**test_params)

def explore_new_architecture():
    for split_num in range(0,10):
        out_dir_meta = '../experiments/caps_heavy_48/'
        num_epochs = 200
        # epoch_start = 0
        # dec_after = ['exp',0.96,3,1e-6]
        # dec_after = ['reduce','max',0.96,5,1e-6]
        dec_after = ['step',100,0.1]

        lr = [0.001]
        pool_type = 'nopool'
        im_size = 48
        model_name = 'caps_for_96'
        save_after = 100
        model_file = None
        # '../experiments/khorrami_caps_k7_s3_new_aug/ck_0_max_500_reduce_max_0.96_20_1e-06_0.001/model_499.pt'
        epoch_start = 0
        type_data = 'single'

        strs_append = '_'.join([str(val) for val in ['all_aug',pool_type,num_epochs]+dec_after+lr])
        out_dir_train = os.path.join(out_dir_meta,'oulu_'+type_data+'_'+str(split_num)+'_'+strs_append)
        print out_dir_train

        train_file = os.path.join('../data/Oulu_CASIA','train_test_files',type_data+'_im','train_'+str(split_num)+'.txt')
        test_file = os.path.join('../data/Oulu_CASIA','train_test_files',type_data+'_im','test_'+str(split_num)+'.txt')
        mean_std_file = os.path.join('../data/Oulu_CASIA','train_test_files',type_data+'_im','train_'+str(split_num)+'_mean_std_val_0_1.npy')
        
        mean_std = np.load(mean_std_file)

        print mean_std

        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((50,50)),
            transforms.RandomCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size,im_size)),
            transforms.ToTensor(),
            transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
            ])


        train_data = dataset.Oulu_Static_Dataset(train_file, data_transforms['train'])
        test_data = dataset.Oulu_Static_Dataset(test_file,  data_transforms['val'])
        
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
                    gpu_id = 2,
                    num_workers = 0,
                    model_file = model_file,
                    epoch_start = epoch_start,
                    network_params = network_params)

        # print train_params
        # param_file = os.path.join(out_dir_train,'params.txt')
        # all_lines = []
        # for k in train_params.keys():
        #     str_print = '%s: %s' % (k,train_params[k])
        #     print str_print
        #     all_lines.append(str_print)
        # util.writeFile(param_file,all_lines)

        # train_model(**train_params)

        test_params = dict(out_dir_train = out_dir_train,
                model_num = num_epochs-1, 
                train_data = train_data,
                test_data = test_data,
                gpu_id = 2,
                model_name = model_name,
                batch_size_val = batch_size_val,
                criterion = 'margin',
                network_params = network_params)
        test_model(**test_params)
        # break


def main():
    # increasing_spread_improving_r1(5e-6,[0.001])
    increasing_spread_improving_r1(0,[0.001],0.5)
    # increasing_spread_improving_r1(0,[0.0001])

    # khorrami_with_val()
    # test_center_crop()
    # test_center_crop()
    # type_data = 'single_im'
    # split_num = 0
    # train_file = os.path.join('../data/Oulu_CASIA','train_test_files',type_data,'train_'+str(split_num)+'.txt')
    # class_weights = util.get_class_weights(util.readLinesFromFile(train_file))
    # print class_weights

    # khorrami_exp_spread()
    # explore_new_architecture()
    # khorrami_new_aug_method()

if __name__=='__main__':
    main()