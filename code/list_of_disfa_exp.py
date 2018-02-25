from train_test_caps import *
from torchvision import datasets, transforms
import models

import os
from helpers import util,visualize,augmenters
import random
import dataset
import numpy as np


def khorrami_margin():
    for split_num in [16]:
    # range(0,10):
        out_dir_meta = '../experiments/disfa/disfa_margin'
        util.makedirs(out_dir_meta)
        
        route_iter = 3
        num_epochs = 20
        epoch_start = 0
        # dec_after = ['exp',0.96,3,1e-6]
        dec_after = ['step',100,0.1]
        # dec_after = ['reduce','min',0.1,50,1e-4]
        # dec_after = ['reduce','max',0.96,5,1e-6]
        # 'reduce':
        #     exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=dec_after[1], factor=dec_after[2], patience=dec_after[3],min_lr=dec_after[4])


        lr = [0,0.01]
        im_resize = 256
        im_size = 224
        model_name = 'vgg_capsule_disfa'
        save_after = 100
        
        train_test_folder = 'train_test_10_6_method'; n_classes = 10;
        
        criterion = nn.MultiLabelMarginLoss()

        
        # MultiLabelSoftMarginLoss()
        # .cuda()
        margin_params = None
        class_weights = None
        model_file = None
        
        strs_append = '_'.join([str(val) for val in [num_epochs]+dec_after+lr])
        out_dir_train = os.path.join(out_dir_meta,'disfa_'+str(split_num)+'_'+strs_append)
        print out_dir_train

        train_file = os.path.join('../data/disfa',train_test_folder,'train_'+str(split_num)+'.txt')
        test_file = os.path.join('../data/disfa',train_test_folder,'test_'+str(split_num)+'.txt')
        mean_std_file = os.path.join('../data/disfa',train_test_folder,'mean_std.npy')
        test_file = train_file
        # mean_std = np.load(mean_std_file)
        mean_std = np.array([[129.1863,104.7624,93.5940],[1.,1.,1.]])
        print mean_std

        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size,im_size)),
            # transforms.RandomCrop(im_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # lambda x: x*255,
            # transforms.Normalize(mean_std[0,:],mean_std[1,:]),
            # lambda x: x[[2, 1, 0],:, : ]
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size,im_size)),
            transforms.ToTensor(),
            # lambda x: x*255,
            # transforms.Normalize(mean_std[0,:],mean_std[1,:]),
            # lambda x: x[ [2, 1, 0],:, :]
            ])
        data_transforms['val_center']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_resize,im_resize)),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            # lambda x: x*255,
            # transforms.Normalize(mean_std[0,:],mean_std[1,:]),
            # lambda x: x[[2, 1, 0],:, : ]
            ])

        train_data = dataset.Disfa_10_6_Dataset(train_file, data_transforms['train'])
        test_data = dataset.Disfa_10_6_Dataset(test_file,  data_transforms['val'])
        test_data_center = dataset.Disfa_10_6_Dataset(test_file,  data_transforms['val_center'])
        
        network_params = dict(n_classes=n_classes, r=route_iter,init=True,class_weights = class_weights)
        
        batch_size = 16
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
                    test_after = 4,
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
        
        test_params['test_data'] = test_data_center
        test_params['post_pend'] = '_center'
        test_model(**test_params)

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


def main():
    khorrami_margin()

if __name__=='__main__':
    main()