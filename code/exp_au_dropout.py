from train_test_caps import *
from torchvision import datasets, transforms
import models

import os
import sys
from helpers import util,visualize,augmenters
import random
import dataset
import numpy as np
import torch
from analysis import getting_accuracy
# from helpers import util,visualize,augmenters
import save_visualizations
import argparse

def get_out_dir_train_name(wdecay,lr,route_iter,fold,model_name='vgg_capsule_bp4d',epoch_stuff=[30,60],res=False, class_weights = False, reconstruct = False, loss_weights = None,exp=False, disfa = False,vgg_base_file = None,vgg_base_file_str = None, mean_file = None, std_file=None, aug_more = False, dropout = 0, gpu_id = 0):
    out_dir_meta = '../experiments_dropout/'+model_name+'_'+str(route_iter)
    num_epochs = epoch_stuff[1]
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    if disfa:
        type_data = 'train_test_8_au_all_method_110_gray_align'
        pre_pend = 'disfa_'+type_data
    else:
        type_data = 'train_test_files_110_gray_align'; n_classes = 12;
        pre_pend = 'bp4d_'+type_data
            
    if aug_more:
        aug_str = 'cropkhAugNoColor'
    else:
        aug_str = 'flip'

    strs_append =  [pre_pend,fold,'reconstruct',reconstruct,aug_str,num_epochs]+dec_after+lr+[dropout]
    
    if loss_weights is not None:
        strs_append += ['lossweights']+loss_weights
    if vgg_base_file_str is not None:
        strs_append += [vgg_base_file_str]

    strs_append = '_'.join([str(val) for val in strs_append])

    out_dir_train =  os.path.join(out_dir_meta,strs_append)
    return out_dir_train, pre_pend,strs_append


def train_gray(wdecay,lr,route_iter,folds=[4,9],model_name='vgg_capsule_bp4d',epoch_stuff=[30,60],res=False, class_weights = False, reconstruct = False, loss_weights = None,exp=False, disfa = False,vgg_base_file = None,vgg_base_file_str = None, mean_file = None, std_file=None, aug_more = False, dropout = 0, gpu_id = 0):
    
    out_dirs = []
    class_weights = True

    out_dir_meta = '../experiments_dropout/'+model_name+'_'+str(route_iter)
    num_epochs = epoch_stuff[1]
    epoch_start = 0
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr
    im_resize = 110
    # 256
    im_size = 96
    save_after = 1
    if disfa:
        dir_files = '../data/disfa'
        # type_data = 'train_test_10_6_method_110_gray_align'; n_classes = 10;
        type_data = 'train_test_8_au_all_method_110_gray_align'; n_classes = 8;
        pre_pend = 'disfa_'+type_data+'_'
        binarize = True
    else:
        dir_files = '../data/bp4d'
        type_data = 'train_test_files_110_gray_align'; n_classes = 12;
        pre_pend = 'bp4d_'+type_data+'_'
        binarize = False
            
    criterion = 'marginmulti'
    criterion_str = criterion

    init = False
    if aug_more:
        aug_str = 'cropflip'
    else:
        aug_str = 'nothing'

    strs_append =  ['reconstruct',reconstruct,aug_str,num_epochs]+dec_after+lr+[dropout]

    if loss_weights is not None:
        strs_append += ['lossweights']+loss_weights
    if vgg_base_file_str is not None:
        strs_append += [vgg_base_file_str]
    
    strs_append = '_'+'_'.join([str(val) for val in strs_append])
    
    
    lr_p=lr[:]
    for split_num in folds:
        
        if res:
            
            strs_appendc = '_'+'_'.join([str(val) for val in ['reconstruct',reconstruct,True,aug_str,criterion_str,init,'wdecay',wdecay,10]+dec_after+lr+['lossweights']+loss_weights+[vgg_base_file_str]])
            
            out_dir_train = os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_appendc)
            model_file = os.path.join(out_dir_train,'model_9.pt')
            epoch_start = 10
            
        else:
            model_file = None    


        margin_params = None
        
        out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
        final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
        if os.path.exists(final_model_file):
            print 'skipping',final_model_file
            continue 
        else:
            print 'not skipping', final_model_file
        
        train_file = os.path.join(dir_files,type_data,'train_'+str(split_num)+'.txt')
        test_file = os.path.join(dir_files,type_data,'test_'+str(split_num)+'.txt')
        if vgg_base_file is None:
            mean_file = os.path.join(dir_files,type_data,'train_'+str(split_num)+'_mean.png')
            std_file = os.path.join(dir_files,type_data,'train_'+str(split_num)+'_std.png')

        print train_file
        print test_file
        print mean_file
        print std_file

        class_weights = util.get_class_weights_au(util.readLinesFromFile(train_file))
        
        data_transforms = {}
        if aug_more:

            resize = None
            print 'AUGING MORE'
            list_of_todos = ['flip']
            # ,'rotate','scale_translate']
            
            data_transforms['train']= transforms.Compose([
                lambda x: augmenters.random_crop(x,im_size),
                lambda x: augmenters.augment_image(x,list_of_todos),
                transforms.ToTensor(),
                lambda x: x*255,
            ])
        else:
            resize = im_size
            data_transforms['train']= transforms.Compose([
                # lambda x: augmenters.random_crop(x,im_size),
                # lambda x: augmenters.horizontal_flip(x),
                transforms.ToTensor(),
                lambda x: x*255,
            ])
        
        data_transforms['val']= transforms.Compose([
            transforms.ToTensor(),
            lambda x: x*255,
        ])

        train_data = dataset.Bp4d_Dataset_Mean_Std_Im(train_file, mean_file, std_file, transform = data_transforms['train'], binarize = binarize, resize = resize)
        test_data = dataset.Bp4d_Dataset_Mean_Std_Im(test_file, mean_file, std_file, resize= im_size, transform = data_transforms['val'], binarize = binarize)

        network_params = dict(n_classes=n_classes,
                                pool_type='max',
                                r=route_iter,
                                init=init,
                                class_weights = class_weights,
                                reconstruct = reconstruct,
                                loss_weights = loss_weights,
                                vgg_base_file = vgg_base_file,
                                dropout = dropout
                                )
        
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
                    plot_after = 100,
                    test_after = 1,
                    lr = lr,
                    dec_after = dec_after, 
                    model_name = model_name,
                    criterion = criterion,
                    gpu_id = gpu_id,
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
                    gpu_id = gpu_id,
                    model_name = model_name,
                    batch_size_val = batch_size_val,
                    criterion = criterion,
                    margin_params = margin_params,
                    network_params = network_params)
        # test_params_train = dict(**test_params)
        # test_params_train['test_data'] = train_data_no_t
        # test_params_train['post_pend'] = '_train'

        print train_params
        param_file = os.path.join(out_dir_train,'params.txt')
        all_lines = []
        for k in train_params.keys():
            str_print = '%s: %s' % (k,train_params[k])
            print str_print
            all_lines.append(str_print)
        util.writeFile(param_file,all_lines)

        # if reconstruct:

        train_model_recon(**train_params)
        test_model_recon(**test_params)
        

    getting_accuracy.print_accuracy(out_dir_meta,pre_pend,strs_append,folds,log='log.txt')




def get_log_val_arr(log_file):
    lines = util.readLinesFromFile(log_file)
    lines = [line for line in lines if 'val accuracy' in line]
    lines = [float(line.split(' ')[-1]) for line in lines]
    return lines

def get_models_accuracy():
    out_dir_meta = '../experiments_dropout'
    exp_name = 'bp4d_norecon'
    wdecay = 0
    lr = [0.001,0.001]
    route_iter = 3
    folds_all = [0,1]
    model_name = 'khorrami_capsule_7_3_gray'
    epoch_stuff = [15,15]
    gpu_id = 0
    # folds = 
    # dropout = 
    commands_all = []

    params_arr = [(0,[2],0),(0,[2],1)]
    # (0,[0,1],0),(0.5,[0,1],1),(0.5,[2],2)]


    xAndYs = []
    legend_strs = []
    for dropout in [0.,0.5]:
        for aug_more in [True,False]:
            val_arr_all = []
            for fold in range(3):
                params = dict(wdecay = wdecay,
                            lr = lr,
                            route_iter = route_iter,
                            model_name = model_name,
                            epoch_stuff = epoch_stuff,
                            gpu_id = gpu_id,
                            fold = fold,
                            dropout = dropout,
                            aug_more = aug_more)
                out_dir_train, pre_pend, post_pend = get_out_dir_train_name(**params)    
                log_file = os.path.join(out_dir_train,'log.txt')
                assert os.path.exists(log_file)
                val_arr = get_log_val_arr(log_file)
                
                val_arr_all.append(val_arr)
            val_arr_all = np.array(val_arr_all)
            val_arr_all = np.mean(val_arr_all,0)
            xAndYs.append((range(len(val_arr_all)),val_arr_all))
            legend_strs.append(' '.join([str(val) for val in [dropout,aug_more]]))

    out_file = os.path.join(out_dir_meta,exp_name+'.jpg')
    visualize.plotSimple(xAndYs,out_file,title = 'Dropout', xlabel='Epoch',ylabel='Val Accuracy',legend_entries = legend_strs, outside = True)







def make_command_str():
    out_dir = '../experiments_dropout'
    # util.mkdir(out_dir)
    exp_name = 'bp4d_notevenflip'
    out_dir_logs = os.path.join(out_dir,exp_name)
    util.mkdir(out_dir_logs)

    out_file_sh = os.path.join(out_dir,exp_name+'.sh')

    wdecay = 0
    lr = [0.001,0.001]
    route_iter = 3
    folds_all = [0,1]
    model_name = 'khorrami_capsule_7_3_gray'
    epoch_stuff = [15,15]
    aug_more = False
    dropout = 0
    gpu_id = 0

    commands_all = []

    params_arr = [(0,[0,1,2],0),(0.5,[0,1,2],1)]
    # (0,[0,1],0),(0.5,[0,1],1),(0.5,[2],2)]
    

    # for dropout in [0,0.5]:
    #     for folds in folds_all:
    for dropout, folds, gpu_id in params_arr:
        out_file = os.path.join(out_dir_logs,'_'.join([str(val) for val in [dropout]+folds])+'.txt')

        command_str = []
        command_str.extend(['python','exp_au_dropout.py'])
        command_str.append('train')
        command_str.extend(['--wdecay', wdecay])
        command_str.extend(['--lr']+lr)
        command_str.extend(['--route_iter', route_iter])
        command_str.extend(['--folds']+ folds)
        command_str.extend(['--model_name', model_name])
        command_str.extend(['--epoch_stuff']+ epoch_stuff)
        if aug_more:
            command_str.extend(['--aug_more'])
        command_str.extend(['--dropout', dropout])
        command_str.extend(['--gpu_id', gpu_id])
        command_str.extend(['>', out_file,'&'])
        command_str = ' '.join([str(val) for val in command_str])
        print command_str
        commands_all.append(command_str)
        # gpu_id +=1

    print out_file_sh
    util.writeFile(out_file_sh, commands_all)

    # params = dict(wdecay = wdecay,
    #             lr=lr,
    #             route_iter = route_iter,
    #             folds= folds,
    #             model_name= model_name,
    #             epoch_stuff=epoch_stuff,
    #             aug_more= aug_more,
    #             dropout = dropout,
    #             gpu_id = gpu_id)
    # train_gray(**params)

def main(args):
    # print args
    # make_command_str()
    # return
    print args
    if len(args)>1 and args[1]=='train':
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--wdecay', metavar='wdecay', type=float, default = 0., help='weight decay')
        parser.add_argument('--lr', metavar='lr', type=float, nargs='+', default = [0.001,0.001], help='learning rate')
        parser.add_argument('--route_iter', metavar='route_iter',default = 3, type=int, help='route_iter')
        parser.add_argument('--folds', metavar='folds', type=int, nargs = '+', default = [0,1,2], help='folds')
        parser.add_argument('--model_name', metavar='model_name', type=str, default = 'khorrami_capsule_7_3_gray', help='model_name')
        parser.add_argument('--epoch_stuff', metavar='epoch_stuff', type=int, default = [15,15], nargs = '+', help='epoch_stuff')
        parser.add_argument('--aug_more', dest='aug_more', default = False, action='store_true', help='aug_more')
        parser.add_argument('--dropout', metavar='dropout', type=float, default = 0., help='dropout')
        parser.add_argument('--gpu_id', metavar='gpu_id', type=int, default = 0, help='gpu_id')
        if len(args)>2:
            args = parser.parse_args(args[2:])
            args = vars(args)
            print args
            train_gray(**args)
    else:
        make_command_str()
        # get_models_accuracy()


    # print args

    # model_name = 'khorrami_capsule_7_3_gray'
    # disfa = False
    # folds = [0,1,2]
    # route_iter = 3
    # epoch_stuff = [15,15]
    # lr = [0.001,0.001]
    # dropout = 0
    # gpu_id = 0
    # aug_more = True
    # train_gray(0,lr=lr,route_iter = route_iter, folds= folds, model_name= model_name, epoch_stuff=epoch_stuff,aug_more= aug_more, dropout = dropout, gpu_id = gpu_id)

    

if __name__=='__main__':
    main(sys.argv)