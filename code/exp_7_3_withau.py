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



def train_khorrami_aug(wdecay,lr,route_iter,folds=[4,9],model_name='vgg_capsule_disfa',epoch_stuff=[30,60],res=False, class_weights = False, reconstruct = False):
    out_dirs = []

    out_dir_meta = '../experiments/'+model_name+str(route_iter)
    num_epochs = epoch_stuff[1]
    epoch_start = 0
    # dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr
    im_resize = 110
    im_size = 96
    save_after = 100
    type_data = 'train_test_files'; n_classes = 8;
    criterion = 'margin'
    criterion_str = criterion

    # criterion = nn.CrossEntropyLoss()
    # criterion_str = 'crossentropy'
    
    init = False
    loss_weights = [1.,0.5,0.5]

    strs_append = '_'+'_'.join([str(val) for val in ['au_sup',loss_weights,'reconstruct',reconstruct,class_weights,'all_aug',criterion_str,init,'wdecay',wdecay,num_epochs]+dec_after+lr])
    pre_pend = 'ck_96_'
    
    lr_p=lr[:]
    for split_num in folds:
        
        if res:

            strs_appendc = '_'+'_'.join([str(val) for val in ['reconstruct',reconstruct,True,'all_aug',criterion_str,init,'wdecay',wdecay,600,'step',600,0.1]+lr_p])
            out_dir_train = os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_appendc)
            model_file = os.path.join(out_dir_train,'model_300.pt')
            epoch_start = 300
            lr =[0.1*lr_curr for lr_curr in lr_p]
        else:
            model_file = None    


        margin_params = None
        
        out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
        final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
        if os.path.exists(final_model_file):
            print 'skipping',final_model_file
            continue 

        train_file = '../data/ck_96/train_test_files/train_emofacscombo_'+str(split_num)+'.txt'
        test_file = '../data/ck_96/train_test_files/test_emofacscombo_'+str(split_num)+'.txt'
        mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
        std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'
        
        mean_im = scipy.misc.imread(mean_file).astype(np.float32)
        std_im = scipy.misc.imread(std_file).astype(np.float32)

        if class_weights:
            print class_weights
            actual_class_weights,au_class_weights = util.get_class_weights(util.readLinesFromFile(train_file),au=True)
            print actual_class_weights
            print au_class_weights
            # actual_class_weights = None
            # au_class_weights = None 
        else:
            actual_class_weights = None
            au_class_weights = None

        # print std_im.shape
        # print np.min(std_im),np.max(std_im)
        # raw_input()

        list_of_to_dos = ['flip','rotate','scale_translate','pixel_augment']
        
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

        train_data = dataset.CK_96_Dataset_WithAU(train_file, mean_file, std_file, data_transforms['train'])
        test_data = dataset.CK_96_Dataset_WithAU(test_file, mean_file, std_file, data_transforms['val'])
        
        network_params = dict(n_classes=n_classes,pool_type='max',r=route_iter,init=init,class_weights = actual_class_weights, reconstruct = reconstruct,au_sup = True, class_weights_au = au_class_weights,loss_weights = loss_weights)

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

        # if reconstruct:
        train_model_recon_au(**train_params)
        # else:
        #     train_model(**train_params)
        #     test_params = dict(out_dir_train = out_dir_train,
        #             model_num = num_epochs-1, 
        #             train_data = train_data,
        #             test_data = test_data,
        #             gpu_id = 0,
        #             model_name = model_name,
        #             batch_size_val = batch_size_val,
        #             criterion = criterion,
        #             margin_params = margin_params,
        #             network_params = network_params)
        #     test_model(**test_params)
        
    getting_accuracy.print_accuracy(out_dir_meta,pre_pend,strs_append,folds,log='log.txt')



def main():
    
    folds = [9]
    # range(10)
    # range(9,10)
    # range(6)
    epoch_stuff = [600,600]
    lr = [0.001,0.001,0.001]
    res = True
    route_iter = 3

    train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3_2class', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = True)
    

    # folds = range(2)+range(3,9)
    # train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = True)

    # train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3', epoch_stuff=epoch_stuff,res=False, class_weights = True)

    # meta_data_dir = 'train_test_files_preprocess_vl'
    # train_khorrami_aug_oulu(0, lr=lr, route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3', epoch_stuff=epoch_stuff,res=False, meta_data_dir = meta_data_dir, class_weights = True)

    # meta_data_dir = 'train_test_files_preprocess_maheen_vl_gray'
    # train_khorrami_aug_oulu(0, lr=lr, route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3', epoch_stuff=epoch_stuff,res=False, meta_data_dir = meta_data_dir, class_weights= True)



if __name__=='__main__':
    main()