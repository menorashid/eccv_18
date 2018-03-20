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

def train_khorrami_aug(wdecay,lr,route_iter,folds=[4,9],model_name='vgg_capsule_disfa',epoch_stuff=[30,60],res=False, class_weights = False, reconstruct = False,loss_weights = None,model_to_test = None,oulu = False):
    
    out_dirs = []
    out_dir_meta = '../experiments/showing_overfitting_justhflip_'+model_name+str(route_iter)
    num_epochs = epoch_stuff[1]
    if model_to_test is None:
        model_to_test = num_epochs -1

    epoch_start = 0
    dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr
    im_resize = 110
    im_size = 96
    save_after = num_epochs 

    if not oulu:
        type_data = 'train_test_files'; n_classes = 8;
        train_pre = os.path.join('../data/ck_96',type_data)
        pre_pend = 'ck_96_'+type_data+'_'
    else:
        type_data = 'three_im_no_neutral_just_strong_False'; n_classes = 6;
        # 'train_test_files'; n_classes = 8;
        train_pre = os.path.join('../data/Oulu_CASIA/train_test_files_preprocess_vl',type_data)
        pre_pend = 'oulu_96_'+type_data+'_'
    

    criterion = 'margin'
    criterion_str = criterion

    init = False
    strs_append_list = ['reconstruct',reconstruct,class_weights,'all_aug',criterion_str,init,'wdecay',wdecay,num_epochs]+dec_after+lr

    if loss_weights is not None:
        strs_append_list = strs_append_list     +['lossweights']+loss_weights
    strs_append = '_'+'_'.join([str(val) for val in strs_append_list])
    
    
    
    lr_p=lr[:]
    for split_num in folds:
        
        model_file = None    
        margin_params = None
        
        out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
        final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
        if os.path.exists(final_model_file):
            print 'skipping',final_model_file
            continue 
        else:
            print 'not skipping', final_model_file
    
        train_file = os.path.join(train_pre,'train_'+str(split_num)+'.txt')
        test_file = os.path.join(train_pre,'test_'+str(split_num)+'.txt')
        mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
        std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')

        mean_im = scipy.misc.imread(mean_file).astype(np.float32)
        std_im = scipy.misc.imread(std_file).astype(np.float32)

        class_weights = util.get_class_weights(util.readLinesFromFile(train_file))


        list_of_to_dos = ['flip']
        
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
        
        network_params = dict(n_classes=n_classes,pool_type='max',r=route_iter,init=init,class_weights = class_weights, reconstruct = reconstruct,loss_weights = loss_weights)
        
        batch_size = 128
        batch_size_val = None
        
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
                    model_num = model_to_test,
                    # num_epochs-1, 
                    train_data = train_data,
                    test_data = test_data,
                    gpu_id = 0,
                    model_name = model_name,
                    batch_size_val = batch_size_val,
                    criterion = criterion,
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


            
        if reconstruct:
            train_model_recon(**train_params)
            # test_model_recon(**test_params)
        else:
            train_model(**train_params)
            # test_model(**test_params)

def main():
    
    lr = [0.001,0.001]
    route_iter = 1
    folds = [9]
    model_name = 'khorrami_capsule_7_3'
    epoch_stuff = [600,600]
    wdecay = 0
    reconstruct = False
    oulu = True
    train_khorrami_aug(wdecay,lr,route_iter,folds,model_name=model_name,epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = reconstruct, oulu = oulu)
    
    lr = [0.001,0.001]
    route_iter = 3
    folds = [9]
    model_name = 'khorrami_capsule_7_3'
    epoch_stuff = [600,600]
    wdecay = 0
    reconstruct = False
    train_khorrami_aug(wdecay,lr,route_iter,folds,model_name=model_name,epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = reconstruct, oulu = oulu)


    lr = [0.001,0.001,0.001]
    route_iter = 3
    folds = [9]
    model_name = 'khorrami_capsule_7_3'
    epoch_stuff = [600,600]
    wdecay = 0
    reconstruct = True
    loss_weights = [1.,1.]
    train_khorrami_aug(wdecay,lr,route_iter,folds,model_name=model_name,epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = reconstruct, oulu = oulu, loss_weights = loss_weights)

    lr = [0.001,0.001,0.001]
    route_iter = 3
    folds = [9]
    model_name = 'khorrami_capsule_7_3'
    epoch_stuff = [600,600]
    wdecay = 0
    reconstruct = True
    loss_weights = [1.,100.]
    train_khorrami_aug(wdecay,lr,route_iter,folds,model_name=model_name,epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = reconstruct, oulu = oulu, loss_weights = loss_weights)

    # folds = range(1,10)

    # lr = [0.001,0.001,0.001]
    # route_iter = 3
    # folds = [0]
    # model_name = 'khorrami_capsule_7_3'
    # epoch_stuff = [600,600]
    # wdecay = 0
    # reconstruct = True
    # loss_weights = [1,100]
    # train_khorrami_aug(wdecay,lr,route_iter,folds,model_name=model_name,epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = reconstruct,loss_weights = loss_weights, oulu = oulu)



if __name__=='__main__':
    main()