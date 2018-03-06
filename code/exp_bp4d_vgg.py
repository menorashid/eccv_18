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



def train_vgg(wdecay,lr,route_iter,folds=[4,9],model_name='vgg_capsule_bp4d',epoch_stuff=[30,60],res=False, class_weights = False, reconstruct = False, loss_weights = None):
    out_dirs = []

    out_dir_meta = '../experiments/'+model_name+str(route_iter)
    num_epochs = epoch_stuff[1]
    epoch_start = 0
    # dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr
    im_resize = 110
    # 256
    im_size = 96
    save_after = 10
    type_data = 'train_test_files_110_color_nodetect'; n_classes = 12;
    criterion = 'marginmulti'
    criterion_str = criterion

    init = False

    strs_append = '_'+'_'.join([str(val) for val in ['reconstruct',reconstruct,class_weights,'all_aug',criterion_str,init,'wdecay',wdecay,num_epochs]+dec_after+lr+['lossweights']+loss_weights])
    
    pre_pend = 'bp4d_256_'
    
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
            continue 
        else:
            print 'not skipping', final_model_file
            # raw_input()
            # continue

        train_file = os.path.join('../data/bp4d',type_data,'train_'+str(split_num)+'.txt')
        test_file = os.path.join('../data/bp4d',type_data,'test_'+str(split_num)+'.txt')

        mean_std = np.array([[129.1863,104.7624,93.5940],[1.,1.,1.]])
        # mean_std = np.array([[93.5940,104.7624,129.1863],[1.,1.,1.]]) #bgr
        print mean_std

        # mean_im = scipy.misc.imread(mean_file).astype(np.float32)
        # std_im = scipy.misc.imread(std_file).astype(np.float32)

        class_weights = util.get_class_weights_au(util.readLinesFromFile(train_file))
        
        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((im_resize,im_resize)),
            transforms.RandomCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            lambda x: x*255,
            transforms.Normalize(mean_std[0,:],mean_std[1,:]),
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size,im_size)),
            transforms.ToTensor(),
            lambda x: x*255,
            transforms.Normalize(mean_std[0,:],mean_std[1,:]),
            ])

        # print train_file
        # print test_file
        train_data = dataset.Bp4d_Dataset(train_file, bgr = False, transform = data_transforms['train'])
        test_data = dataset.Bp4d_Dataset(test_file, bgr = False, transform = data_transforms['val'])
        
        network_params = dict(n_classes=n_classes,pool_type='max',r=route_iter,init=init,class_weights = class_weights, reconstruct = reconstruct,loss_weights = loss_weights)
        
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

        # if reconstruct:
        train_model_recon(**train_params)
        # else:
        #     train_model(**train_params)
        # test_params = dict(out_dir_train = out_dir_train,
        #         model_num = num_epochs-1, 
        #         train_data = train_data,
        #         test_data = test_data,
        #         gpu_id = 0,
        #         model_name = model_name,
        #         batch_size_val = batch_size_val,
        #         criterion = criterion,
        #         margin_params = margin_params,
        #         network_params = network_params)
        # test_model(**test_params)
        
    getting_accuracy.print_accuracy(out_dir_meta,pre_pend,strs_append,folds,log='log.txt')


def main():
    
    folds = range(3)
    
    epoch_stuff = [100,100]
    lr = [0.001,0.001,0.001]
    route_iter = 3

    train_vgg(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3_color', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = False, loss_weights = [1.,1.])

    # meta_data_dir = 'train_test_files_preprocess_maheen_vl_gray'
    # train_khorrami_aug(0,lr=lr,route_iter = route_iter, folds= folds, model_name='khorrami_capsule_7_3', epoch_stuff=epoch_stuff,res=False, class_weights = True, reconstruct = True,oulu= oulu, meta_data_dir = meta_data_dir)




if __name__=='__main__':
    main()