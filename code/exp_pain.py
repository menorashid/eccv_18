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
import cv2
import itertools


def train_with_vgg(lr,
                route_iter,
                train_file_pre,
                test_file_pre,
                out_dir_pre,
                n_classes,
                folds = [4,9],
                model_name='vgg_capsule_disfa',
                epoch_stuff=[30,
                60],
                res=False,
                reconstruct = False,
                loss_weights = None,
                exp = False,
                dropout = 0,
                gpu_id = 0,
                aug_more = 'flip',
                model_to_test = None,
                save_after = 1,
                batch_size = 32,
                batch_size_val = 32,
                criterion = 'marginmulti'):

    # torch.setdefaulttensortype('torch.FloatTensor')

    num_epochs = epoch_stuff[1]
    
    if model_to_test is None:
        model_to_test = num_epochs -1

    epoch_start = 0
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr
    im_resize = 256
    im_size = 224
    model_file = None
    margin_params = None

    for split_num in folds:
        # post_pend = [split_num,'reconstruct',reconstruct]+aug_more+[num_epochs]+dec_after+lr+[dropout]
        # out_dir_train =  '_'.join([str(val) for val in [out_dir_pre]+post_pend]);
        out_dir_train = get_out_dir_train_name(out_dir_pre,lr,route_iter,split_num,epoch_stuff,reconstruct, exp, dropout, aug_more,loss_weights)

        print out_dir_train
        # raw_input()

        final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
        if os.path.exists(final_model_file):
            print 'skipping',final_model_file
            # continue 
        else:
            print 'not skipping', final_model_file
        
        train_file = train_file_pre+str(split_num)+'.txt'
        test_file = test_file_pre+str(split_num)+'.txt'

        class_weights = util.get_class_weights_au(util.readLinesFromFile(train_file))
        # class_weights = None


    
        mean_std = np.array([[93.5940,104.7624,129.1863],[1.,1.,1.]]) #bgr
        std_div = np.array([0.225*255,0.224*255,0.229*255])
        bgr= True
    
        list_of_to_dos = aug_more
        print list_of_to_dos
        
            

        data_transforms = {}
        train_resize = None
        list_transforms = []
        if 'hs' in list_of_to_dos:
            print '**********HS!!!!!!!'
            list_transforms.append(lambda x: augmenters.random_crop(x,im_size))
            list_transforms.append(lambda x: augmenters.hide_and_seek(x))
            if 'flip' in list_of_to_dos:
                list_transforms.append(lambda x: augmenters.horizontal_flip(x))
            list_transforms.append(transforms.ToTensor())
        elif 'flip' in list_of_to_dos and len(list_of_to_dos)==1:
            train_resize=im_size
            list_transforms.extend([lambda x: augmenters.horizontal_flip(x),
                                    transforms.ToTensor()])
        elif 'none' in list_of_to_dos:
            train_resize=im_size
            list_transforms.append(transforms.ToTensor())

            # data_transforms['train']= transforms.Compose([
            #     # lambda x: augmenters.random_crop(x,im_size),
            #     transforms.ToTensor(),
            # ])
        else:
            # data_transforms['train']= transforms.Compose([
            list_transforms.append(lambda x: augmenters.random_crop(x,im_size))
            list_transforms.append(lambda x: augmenters.augment_image(x,list_of_to_dos,color=True,im_size = im_size))
            list_transforms.append(transforms.ToTensor())
                # lambda x: x*255.
            # ])

        list_transforms_val = [transforms.ToTensor()]
        
        if torch.version.cuda.startswith('9.1'):
            list_transforms.append(lambda x: x.float())
        else:
            list_transforms.append(lambda x: x*255.)
            
        
        data_transforms['train']= transforms.Compose(list_transforms)
        data_transforms['val']= transforms.Compose(list_transforms_val)


        train_data = dataset.Bp4d_Dataset_with_mean_std_val(train_file, bgr = bgr, binarize = True, mean_std = mean_std, transform = data_transforms['train'],resize=train_resize)
        test_data = dataset.Bp4d_Dataset_with_mean_std_val(test_file, bgr = bgr, binarize= True, mean_std = mean_std, transform = data_transforms['val'], resize = im_size)
    
        if 'dropout' in model_name:
            network_params = dict(n_classes = n_classes, pool_type = 'max', r = route_iter, init = False , class_weights = class_weights, reconstruct = reconstruct, loss_weights = loss_weights, std_div = std_div, dropout = dropout)
        else:
            network_params = dict(n_classes = n_classes, pool_type = 'max', r = route_iter, init = False , class_weights = class_weights, reconstruct = reconstruct, loss_weights = loss_weights, std_div = std_div)
            
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
                    weight_decay = 0)
        test_params = dict(out_dir_train = out_dir_train,
                    model_num = model_to_test, 
                    train_data = train_data,
                    test_data = test_data,
                    gpu_id = gpu_id,
                    model_name = model_name,
                    batch_size_val = batch_size_val,
                    criterion = criterion,
                    margin_params = margin_params,
                    network_params = network_params,
                    post_pend = '',
                    barebones=True)
        
        print train_params
        param_file = os.path.join(out_dir_train,'params.txt')
        all_lines = []
        for k in train_params.keys():
            str_print = '%s: %s' % (k,train_params[k])
            print str_print
            all_lines.append(str_print)
        
        train_model_recon(**train_params)
        test_model_recon(**test_params)
        






def get_out_dir_train_name(out_dir_pre,lr,route_iter,fold,epoch_stuff=[30,60],reconstruct = False, exp = False, dropout = 0, aug_more = ['flip'],loss_weights = None):
    
    num_epochs = epoch_stuff[1]
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]


    post_pend = [fold,'reconstruct',reconstruct]+aug_more+[num_epochs]+dec_after+lr+[dropout]

    if loss_weights is not None:
        post_pend = post_pend+['loss_weights']+loss_weights

    out_dir_train =  '_'.join([str(val) for val in [out_dir_pre]+post_pend]);

    return out_dir_train






def make_command_str():
    out_dir = '../experiments_pain'

    train_file_pre = '../data/pain/train_test_files_loo_1_thresh_au_only/train_'
    test_file_pre =  '../data/pain/train_test_files_loo_1_thresh_au_only/test_'
    util.mkdir(out_dir)
    exp_name = 'pain_train_1_thresh'
    out_dir_logs = os.path.join(out_dir,exp_name)
    util.mkdir(out_dir_logs)

    out_file_sh = os.path.join(out_dir,exp_name+'.sh')

    wdecay = 0
    route_iter = 3
    n_classes = 6
    model_name = 'vgg_capsule_7_3_with_dropout'
    epoch_stuff = [350,20]
    aug_more = [['flip','rotate','scale_translate']]
    folds = [[0]]
    exp = True
    reconstruct = True
    batch_size_val = 32
    batch_size = 32

    dropout = [0]
    lr_meta = [[0.0001,0.001,0.001]]
    loss_weights = [1.,1.]

    commands_all = []

    params_arr = itertools.product(aug_more,dropout,folds,lr_meta)
    params_arr = [tuple(list(tups)+[val]) for val,tups in enumerate(params_arr)]
    for p in params_arr:
        print p

    for aug_more, dropout, folds, lr, gpu_id in params_arr:
        out_file = os.path.join(out_dir_logs,'_'.join([str(val) for val in aug_more+[dropout]+folds])+'.txt')

        out_dir_pre = os.path.join(out_dir,model_name+'_'+str(route_iter),'au_only_1_pain_thresh_rerun')
        
        command_str = []
        command_str.extend(['python','exp_pain.py'])
        command_str.append('train')
        command_str.extend(['--lr']+lr)
        command_str.extend(['--route_iter', route_iter])
        
        command_str.extend(['--out_dir_pre', out_dir_pre])
        command_str.extend(['--train_file_pre', train_file_pre])
        command_str.extend(['--test_file_pre', test_file_pre])
        command_str.extend(['--n_classes',n_classes])

        command_str.extend(['--batch_size', batch_size])
        command_str.extend(['--batch_size_val', batch_size_val])

        command_str.extend(['--folds']+ folds)
        command_str.extend(['--model_name', model_name])
        command_str.extend(['--epoch_stuff']+ epoch_stuff)
        command_str.extend(['--aug_more']+aug_more)
        command_str.extend(['--dropout', dropout])
        command_str.extend(['--gpu_id', gpu_id])

        command_str.extend(['--loss_weights']+loss_weights)

        if reconstruct:
            command_str.extend(['--reconstruct'])
        if exp:
            command_str.extend(['--exp'])

        command_str.extend(['>', out_file,'&'])
        command_str = ' '.join([str(val) for val in command_str])
        print command_str
        commands_all.append(command_str)

    print out_file_sh
    util.writeFile(out_file_sh, commands_all)


def check():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print torch.__version__
    a = 255*np.ones((100,100,3));
    print type(a[0][0][0])
    print a.shape
    ts = transforms.Compose([
                transforms.ToTensor()
                ])
    output = ts(a)
    print output.size(),torch.min(output),torch.max(output),type(output)


def main(args):


    # check()
    # return
    if len(args)>1 and args[1]=='train':
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--lr', metavar='lr', type=float, nargs='+', default = [0.001,0.001], help='learning rate')
        parser.add_argument('--route_iter', metavar='route_iter',default = 3, type=int, help='route_iter')
        parser.add_argument('--train_file_pre', metavar='train_file_pre',default = '', type=str, help='train_file_pre')
        parser.add_argument('--test_file_pre', metavar='test_file_pre',default = '', type=str, help='test_file_pre')
        parser.add_argument('--out_dir_pre', metavar='out_dir_pre',default = '', type=str, help='out_dir_pre')
        parser.add_argument('--n_classes', metavar='n_classes',default = 0, type=int, help='n_classes')

        parser.add_argument('--folds', metavar='folds', type=int, nargs = '+', default = [0,1,2], help='folds')
        parser.add_argument('--model_name', metavar='model_name', type=str, default = 'khorrami_capsule_7_3_gray', help='model_name')
        parser.add_argument('--epoch_stuff', metavar='epoch_stuff', type=int, default = [15,15], nargs = '+', help='epoch_stuff')
        parser.add_argument('--aug_more', dest='aug_more', nargs='+',default = ['flip'], type=str, help='aug_more')
        parser.add_argument('--dropout', metavar='dropout', type=float, default = 0., help='dropout')
        parser.add_argument('--gpu_id', metavar='gpu_id', type=int, default = 0, help='gpu_id')
        parser.add_argument('--reconstruct', dest='reconstruct', default = False, action='store_true', help='reconstruct')
        parser.add_argument('--batch_size', metavar='batch_size', default = 32, type=int, help='batch_size')
        parser.add_argument('--batch_size_val', metavar='batch_size_val', default = 32, type=int, help='batch_size_val')
        parser.add_argument('--exp', dest='exp', default = False, action='store_true', help='exp')
        parser.add_argument('--loss_weights', dest='loss_weights', default = [1.,1.],nargs = '+', type = float,help='loss_weights')
        

        if len(args)>2:
            args = parser.parse_args(args[2:])
            args = vars(args)
            print args
            train_with_vgg(**args)

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