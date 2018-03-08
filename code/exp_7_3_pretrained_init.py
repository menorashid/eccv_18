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


def save_conv_features():
    num_splits = 10

    pretrained_net_dir_pre = '../experiments/khorrami_ck_96_caps_bl/ck_'
    pretrained_net_dir_post = os.path.join('_khorrami_ck_96_300_exp_0.96_100_1e-06_0.001_0.001','model_299.pt')
    out_dir_conv = '../experiments/khorrami_ck_96_caps_bl_just_conv'
    util.mkdir(out_dir_conv)
    for split_num in range(10):
        model_curr = pretrained_net_dir_pre+str(split_num)+pretrained_net_dir_post
        model_curr = torch.load(model_curr)
        model_curr = list(model_curr.features)[:6]
        model_curr = nn.Sequential(*model_curr)
        model_curr = model_curr.cpu()
        # arrs = [model[0].weight.data.numpy(),model[0].bias.data.numpy(),model[3].weight.data.numpy(),model[3].bias.data.numpy()]
        out_file = os.path.join(out_dir_conv,'model_'+str(split_num)+'.pt')
        print out_file
        torch.save(model_curr,out_file)
        # np.savez(out_file,arrs)
        # torch.save(model_curr,out_file)




def train_khorrami_aug(wdecay,lr,route_iter,folds=[4,9],model_name='vgg_capsule_disfa',epoch_stuff=[30,60],res=False, class_weights = False, reconstruct = False, oulu = False, meta_data_dir = None,loss_weights = None):
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
    if oulu:
        type_data = 'three_im_no_neutral_just_strong_False'; n_classes = 6;
    criterion = 'margin'
    criterion_str = criterion

    pretrained_net_dir_pre ='../experiments/khorrami_ck_96_caps_bl_just_conv/model_';
    pretrained_net_dir_post = '.pt'
    
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
            # raw_input()
            # continue 
        else:
            print 'not skipping', final_model_file
            # raw_input()
            # continue

        if not oulu:
            train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
            test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
            mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
            std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'
            net_model_file = pretrained_net_dir_pre+str(split_num)+pretrained_net_dir_post
        else:
            train_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'.txt')
            test_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'test_'+str(split_num)+'.txt')
            mean_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'_mean.png')
            std_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'_std.png')
            net_model_file = pretrained_net_dir_pre+str(split_num)+pretrained_net_dir_post

        mean_im = scipy.misc.imread(mean_file).astype(np.float32)
        std_im = scipy.misc.imread(std_file).astype(np.float32)

        class_weights = util.get_class_weights(util.readLinesFromFile(train_file))

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

        # train_data = dataset.CK_96_Dataset_Just_Mean(train_file, mean_file, data_transforms['train'])
        # test_data = dataset.CK_96_Dataset_Just_Mean(test_file, mean_file, data_transforms['val'])
        train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
        train_data_no_t = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['val'])
        test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
        
        network_params = dict(n_classes=n_classes,model_file=net_model_file,r=route_iter,init=init,class_weights = class_weights, reconstruct = reconstruct,loss_weights = loss_weights)
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
        str_print = '%s: %s' % ('augmenters',list_of_to_dos)
        print str_print
        util.writeFile(param_file,all_lines)


            
        if reconstruct:
            train_model_recon(**train_params)
            test_model_recon(**test_params)
            test_model_recon(**test_params_train)

        else:
            train_model(**train_params)
            test_model(**test_params)

        
    # getting_accuracy.print_accuracy(out_dir_meta,pre_pend,strs_append,folds,log='log.txt')
    # getting_accuracy.view_loss_curves(out_dir_meta,pre_pend,strs_append,folds,num_epochs-1)


def main():
    # save_conv_features()
    # return
    folds = range(10)
    
    epoch_stuff = [300,300]
    # [600,600]
    lr = [0,0.001,0.01]
    # res = True
    route_iter = 3

    train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3_bigclass_pretrained', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = True)

    # train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3_bigclass', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = True)

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