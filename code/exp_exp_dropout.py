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


def get_out_dir_train_name(wdecay,lr,route_iter,fold,model_name='vgg_capsule_disfa',epoch_stuff=[30,60],res=False, reconstruct = False, oulu = False, meta_data_dir = 'train_test_files_preprocess_vl',loss_weights = None, exp = False, dropout = 0, gpu_id = 0, aug_more = 'flip'):
    
    out_dir_meta = '../experiments_dropout/'+model_name+'_'+str(route_iter)
    num_epochs = epoch_stuff[1]
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    type_data = 'train_test_files'; n_classes = 8;
    train_pre = os.path.join('../data/ck_96',type_data)
    test_pre =  os.path.join('../data/ck_96',type_data)

    if oulu:
        type_data = 'three_im_no_neutral_just_strong_False'; n_classes = 6;

    if oulu:
        pre_pend = 'oulu_96_'+meta_data_dir
    else:
        pre_pend = 'ck_96_'+type_data


    criterion = 'margin'
    criterion_str = criterion

    strs_append_list = [pre_pend,fold,'reconstruct',reconstruct,aug_more,num_epochs]+dec_after+lr+[dropout]

    if loss_weights is not None:
        strs_append_list = strs_append_list     +['lossweights']+loss_weights
    
    strs_append = '_'.join([str(val) for val in strs_append_list])
        
    out_dir_train =  os.path.join(out_dir_meta,strs_append)
    return out_dir_train

def train_gray(wdecay,lr,route_iter,folds = [4,9],model_name='vgg_capsule_disfa',epoch_stuff=[30,60],res=False, reconstruct = False, oulu = False, meta_data_dir = 'train_test_files_preprocess_vl',loss_weights = None, exp = False, dropout = 0, gpu_id = 0, aug_more = 'flip', model_to_test = None):


    out_dir_meta = '../experiments_dropout/'+model_name+'_'+str(route_iter)
    num_epochs = epoch_stuff[1]
    if model_to_test is None:
        model_to_test = num_epochs -1

    epoch_start = 0
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr
    im_resize = 110
    im_size = 96
    save_after = 100

    type_data = 'train_test_files'; n_classes = 8;
    train_pre = os.path.join('../data/ck_96',type_data)
    test_pre =  os.path.join('../data/ck_96',type_data)

    if oulu:
        type_data = 'three_im_no_neutral_just_strong_False'; n_classes = 6;
    criterion = 'margin'
    criterion_str = criterion

    
    init = False
    strs_append_list = ['reconstruct',reconstruct]+aug_more+[num_epochs]+dec_after+lr+[dropout]

    if loss_weights is not None:
        strs_append_list = strs_append_list     +['lossweights']+loss_weights
    strs_append = '_'+'_'.join([str(val) for val in strs_append_list])
    
    if oulu:
        pre_pend = 'oulu_96_'+meta_data_dir+'_'
    else:
        pre_pend = 'ck_96_'+type_data+'_'
    
    lr_p=lr[:]
    for split_num in folds:
        
        if res:
            print 'what to res?'
            raw_input()
        else:
            model_file = None    


        margin_params = None
        
        out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
        print out_dir_train
        
        final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
        if os.path.exists(final_model_file):
            print 'skipping',final_model_file
            # raw_input()
            continue 
        else:
            print 'not skipping', final_model_file
            # raw_input()
            # continue

        if not oulu:
            train_file = os.path.join(train_pre,'train_'+str(split_num)+'.txt')
            test_file_easy = os.path.join(train_pre,'test_'+str(split_num)+'.txt')
            test_file = os.path.join(test_pre,'test_'+str(split_num)+'.txt')
            mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
            std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')

        else:
            train_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'.txt')
            test_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'test_'+str(split_num)+'.txt')
            mean_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'_mean.png')
            std_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'_std.png')

        mean_im = scipy.misc.imread(mean_file).astype(np.float32)
        std_im = scipy.misc.imread(std_file).astype(np.float32)

        class_weights = util.get_class_weights(util.readLinesFromFile(train_file))


        list_of_to_dos = aug_more
        print list_of_to_dos
        # raw_input()
        # aug_more.split('_')
        # ['flip','rotate','scale_translate', 'pixel_augment']
        
        data_transforms = {}
        if 'hs' in list_of_to_dos:
            print '**********HS!!!!!!!'
            list_transforms = [lambda x: augmenters.hide_and_seek(x)]
            if 'flip' in list_of_to_dos:
                list_transforms.append(lambda x: augmenters.horizontal_flip(x))
            list_transforms = list_transforms+ [transforms.ToTensor(),lambda x: x*255.]
            print list_transforms
            data_transforms['train']= transforms.Compose(list_transforms)
        elif 'none' in list_of_to_dos:
            print 'DOING NOTHING!!!!!!'
            data_transforms['train']= transforms.Compose([
                transforms.ToTensor(),
                lambda x: x*255.
            ])
        else:
            data_transforms['train']= transforms.Compose([
                lambda x: augmenters.augment_image(x,list_of_to_dos,mean_im,std_im,im_size),
                transforms.ToTensor(),
                lambda x: x*255.
            ])
        
        data_transforms['val']= transforms.Compose([
            transforms.ToTensor(),
            lambda x: x*255.
            ])

        print data_transforms['train']
        # raw_input()

        # train_data = dataset.CK_96_Dataset_Just_Mean(train_file, mean_file, data_transforms['train'])
        # test_data = dataset.CK_96_Dataset_Just_Mean(test_file, mean_file, data_transforms['val'])

        print train_file
        print test_file
        print std_file
        print mean_file
        # raw_input()

        train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
        test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
        
        network_params = dict(n_classes=n_classes,pool_type='max',r=route_iter,init=init,class_weights = class_weights, reconstruct = reconstruct,loss_weights = loss_weights, dropout = dropout)
    
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
                    model_num = model_to_test,
                    # num_epochs-1, 
                    train_data = train_data,
                    test_data = test_data,
                    gpu_id = gpu_id,
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


       # if reconstruct:
        train_model_recon(**train_params)
        test_model_recon(**test_params)
        # else:
        #     train_model(**train_params)
        #     test_model(**test_params)


        
    getting_accuracy.print_accuracy(out_dir_meta,pre_pend,strs_append,folds,log='log.txt')
    # getting_accuracy.view_loss_curves(out_dir_meta,pre_pend,strs_append,folds,num_epochs-1)



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



def checking_aug(wdecay,lr,route_iter,folds = [4,9],model_name='vgg_capsule_disfa',epoch_stuff=[30,60],res=False, reconstruct = False, oulu = False, meta_data_dir = 'train_test_files_preprocess_vl',loss_weights = None, exp = False, dropout = 0, gpu_id = 0, aug_more = 'flip', model_to_test = None):


    out_dir_meta = '../experiments_dropout/'+model_name+'_'+str(route_iter)
    num_epochs = epoch_stuff[1]
    if model_to_test is None:
        model_to_test = num_epochs -1

    epoch_start = 0
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr
    im_resize = 110
    im_size = 96
    save_after = 100

    type_data = 'train_test_files'; n_classes = 8;
    train_pre = os.path.join('../data/ck_96',type_data)
    test_pre =  os.path.join('../data/ck_96',type_data)

    if oulu:
        type_data = 'three_im_no_neutral_just_strong_False'; n_classes = 6;
    criterion = 'margin'
    criterion_str = criterion

    
    init = False
    strs_append_list = ['reconstruct',reconstruct]+aug_more+[num_epochs]+dec_after+lr+[dropout]

    if loss_weights is not None:
        strs_append_list = strs_append_list     +['lossweights']+loss_weights
    strs_append = '_'+'_'.join([str(val) for val in strs_append_list])
    
    if oulu:
        pre_pend = 'oulu_96_'+meta_data_dir+'_'
    else:
        pre_pend = 'ck_96_'+type_data+'_'
    
    lr_p=lr[:]
    for split_num in folds:
        
        if res:
            print 'what to res?'
            raw_input()
        else:
            model_file = None    


        margin_params = None
        
        out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
        print out_dir_train
        
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
            train_file = os.path.join(train_pre,'train_'+str(split_num)+'.txt')
            # train_file = os.path.join(train_pre,'test_'+str(split_num)+'.txt')
            test_file_easy = os.path.join(train_pre,'test_'+str(split_num)+'.txt')
            test_file = os.path.join(test_pre,'test_'+str(split_num)+'.txt')
            mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
            std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')

        else:
            train_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'.txt')
            test_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'test_'+str(split_num)+'.txt')
            mean_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'_mean.png')
            std_file = os.path.join('../data/Oulu_CASIA',meta_data_dir, type_data, 'train_'+str(split_num)+'_std.png')

        mean_im = scipy.misc.imread(mean_file).astype(np.float32)
        std_im = scipy.misc.imread(std_file).astype(np.float32)

        class_weights = util.get_class_weights(util.readLinesFromFile(train_file))


        list_of_to_dos = aug_more
        print list_of_to_dos
        # raw_input()
        # aug_more.split('_')
        # ['flip','rotate','scale_translate', 'pixel_augment']
        
        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            # lambda x: augmenters.augment_image(x,list_of_to_dos,mean_im,std_im,im_size),
            lambda x: augmenters.hide_and_seek(x, div_sizes = [9,7,5,3], hide_prob = 0.5,fill_val = 0),
            transforms.ToTensor(),
            lambda x: x*255.
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToTensor(),
            lambda x: x*255.
            ])

        # train_data = dataset.CK_96_Dataset_Just_Mean(train_file, mean_file, data_transforms['train'])
        # test_data = dataset.CK_96_Dataset_Just_Mean(test_file, mean_file, data_transforms['val'])

        print train_file
        print test_file
        print std_file
        print mean_file
        # raw_input()



        train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])

        batch_size = 1
        # len(train_data)
        print batch_size
        
        train_dataloader = torch.utils.data.DataLoader(train_data, 
                        batch_size=batch_size,
                        shuffle=False)
        
        out_dir_im = '../experiments_dropout/checking_aug/im_flip_check'
        util.makedirs(out_dir_im)

        for num_iter_train,batch in enumerate(train_dataloader):
            if num_iter_train%100==0:
                print num_iter_train

            ims = batch['image'].cpu().numpy()
            # print ims.shape
            # print np.mean(ims)
            # print np.std(ims)
            # print np.min(ims),np.max(ims)
            # print np.min(train_data.mean),np.max(train_data.mean)
            # print np.min(train_data.std),np.max(train_data.std)
            # continue
            labels = batch['label']

            # ims = ims*train_data.std[np.newaxis,np.newaxis,:,:]
            # ims = ims+train_data.mean[np.newaxis,np.newaxis,:,:]

            for num_curr, im_curr in enumerate(ims):
                if num_curr%100==0:
                    print num_curr
                im_curr = im_curr.squeeze()
                # print np.min(im_curr),np.max(im_curr)

                # print im_curr.shape
                out_file_curr = os.path.join(out_dir_im, '_'.join([str(val) for val in [num_iter_train,num_curr]])+'.png')
                # print out_file_curr
                # print np.min(im_curr),np.max(im_curr)
                # raw_input()
                # cv2.imwrite(out_file_curr,im_curr)
                scipy.misc.imsave(out_file_curr,im_curr)
                # break
            # break


            # print ims.shape
            # print train_data.mean.shape
            # print train_data.std.shape

        visualize.writeHTMLForFolder(out_dir_im,'.png')


    
        print 'done'




def make_command_str():
    out_dir = '../experiments_dropout'
    # util.mkdir(out_dir)
    exp_name = 'oulu_3_none'
    out_dir_logs = os.path.join(out_dir,exp_name)
    util.mkdir(out_dir_logs)

    out_file_sh = os.path.join(out_dir,exp_name+'.sh')

    wdecay = 0
    lr = [0.001,0.001]
    route_iter = 3
    # folds_all = [4,9]
    model_name = 'khorrami_capsule_7_3_bigclass_with_dropout'
    epoch_stuff = [600,600]
    aug_more = [['none']]
    # [['hs','flip']]
    # ,['flip']]
    # l[i:i+n] for i in xrange(0, len(l), n)
    folds = [0,1,2,3,5,6,7,8]
    folds = [folds[i:i+3] for i in range(0,len(folds),3)]
    # print folds
    # raw_input()
    dropout = [0,0.5]


    # ['flip','rotate']
    # ,'scale_translate']
    oulu = True
    # ['flip']
    # dropout = 0
    # gpu_id = 0

    commands_all = []

    # params_arr = [(0,[4],0),(0.5,[4],1),(0,[9],2),(0.5,[9],3)]

    params_arr = itertools.product(aug_more,dropout,folds)
    params_arr = [tuple(list(tups)+[val]) for val,tups in enumerate(params_arr)]
    for p in params_arr:
        print p

    # return
    # [(0.5,[4],2),(0.5,[9],3)]
    # ,(0.5,[4],1),(0.5,[9],3)]
    # [(0.6,[9],0),(0.7,[9],1),(0.8,[9],2),(0.9,[9],3)]
    
    for aug_more, dropout, folds, gpu_id in params_arr:
        out_file = os.path.join(out_dir_logs,'_'.join([str(val) for val in aug_more+[dropout]+folds])+'.txt')

        command_str = []
        command_str.extend(['python','exp_exp_dropout.py'])
        command_str.append('train')
        command_str.extend(['--wdecay', wdecay])
        command_str.extend(['--lr']+lr)
        command_str.extend(['--route_iter', route_iter])
        command_str.extend(['--folds']+ folds)
        command_str.extend(['--model_name', model_name])
        command_str.extend(['--epoch_stuff']+ epoch_stuff)
        # if aug_more:
        command_str.extend(['--aug_more']+aug_more)
        command_str.extend(['--dropout', dropout])
        command_str.extend(['--gpu_id', gpu_id])
        if oulu:
            command_str.extend(['--oulu'])
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
    # # make_command_str()
    # model_name = 'khorrami_capsule_7_3_bigclass_with_dropout'
    # dir_name = get_out_dir_train_name(0,[0.001,0.001],3,4,model_name=model_name,epoch_stuff=[30,60],res=False, reconstruct = False, oulu = False, meta_data_dir = None,loss_weights = None, exp = False, dropout = 0, gpu_id = 0, aug_more = 'flip')
    # train_gray(0,[0.001,0.001],3,[4],model_name=model_name,epoch_stuff=[30,60],res=False, reconstruct = False, oulu = False, meta_data_dir = None,loss_weights = None, exp = False, dropout = 0, gpu_id = 0, aug_more = 'flip')

    # # (wdecay,lr,route_iter,folds = [4,9],model_name='vgg_capsule_disfa',epoch_stuff=[30,60],res=False, reconstruct = False, oulu = False, meta_data_dir = None,loss_weights = None, exp = False, dropout = 0, gpu_id = 0, aug_more = 'flip', model_to_test = None)
    # print dir_name

    # return
    # print args
    if len(args)>1 and args[1]=='train':
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--wdecay', metavar='wdecay', type=float, default = 0., help='weight decay')
        parser.add_argument('--lr', metavar='lr', type=float, nargs='+', default = [0.001,0.001], help='learning rate')
        parser.add_argument('--route_iter', metavar='route_iter',default = 3, type=int, help='route_iter')
        parser.add_argument('--folds', metavar='folds', type=int, nargs = '+', default = [0,1,2], help='folds')
        parser.add_argument('--model_name', metavar='model_name', type=str, default = 'khorrami_capsule_7_3_gray', help='model_name')
        parser.add_argument('--epoch_stuff', metavar='epoch_stuff', type=int, default = [15,15], nargs = '+', help='epoch_stuff')
        # parser.add_argument('--aug_more', dest='aug_more', default = 'flip', type=str, help='aug_more')
        parser.add_argument('--aug_more', dest='aug_more', nargs='+',default = ['flip'], type=str, help='aug_more')
        parser.add_argument('--dropout', metavar='dropout', type=float, default = 0., help='dropout')
        parser.add_argument('--gpu_id', metavar='gpu_id', type=int, default = 0, help='gpu_id')
        # parser.add_argument('--oulu', metavar='gpu_id', type=int, default = 0, help='gpu_id')
        parser.add_argument('--oulu', dest='oulu', default=False, action='store_true')
        if len(args)>2:
            args = parser.parse_args(args[2:])
            args = vars(args)
            print args
            train_gray(**args)
    elif len(args)>1 and args[1]=='debug':
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--wdecay', metavar='wdecay', type=float, default = 0., help='weight decay')
        parser.add_argument('--lr', metavar='lr', type=float, nargs='+', default = [0.001,0.001], help='learning rate')
        parser.add_argument('--route_iter', metavar='route_iter',default = 3, type=int, help='route_iter')
        parser.add_argument('--folds', metavar='folds', type=int, nargs = '+', default = [0,1,2], help='folds')
        parser.add_argument('--model_name', metavar='model_name', type=str, default = 'khorrami_capsule_7_3_gray', help='model_name')
        parser.add_argument('--epoch_stuff', metavar='epoch_stuff', type=int, default = [15,15], nargs = '+', help='epoch_stuff')
        # parser.add_argument('--aug_more', dest='aug_more', default = 'flip', type=str, help='aug_more')
        parser.add_argument('--aug_more', dest='aug_more', nargs='+',default = ['flip'], type=str, help='aug_more')
        parser.add_argument('--dropout', metavar='dropout', type=float, default = 0., help='dropout')
        parser.add_argument('--gpu_id', metavar='gpu_id', type=int, default = 0, help='gpu_id')
        # parser.add_argument('--oulu', metavar='gpu_id', type=int, default = 0, help='gpu_id')
        parser.add_argument('--oulu', dest='oulu', default=False, action='store_true')
        args = parser.parse_args(args[2:])
        args = vars(args)
        print args
        checking_aug(**args)

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