from train_test_caps import *
from torchvision import datasets, transforms
import models

import os
from helpers import util,visualize,augmenters
import random
import dataset
import numpy as np
from analysis import getting_accuracy

def khorrami_bl_exp(mmi= False, model_to_test = None):


    out_dir_meta = '../experiments/khorrami_ck_96_caps_bl/'
    # pre_pend = os.path.join(out_dir_meta,'ck_')
    # post_pend = strs_append


    num_epochs = 300
    epoch_start = 0
    # dec_after = ['exp',0.96,350,1e-6]
    dec_after = ['exp',0.96,350,1e-6]
    # dec_after = ['step',num_epochs,0.1]
    lr = [0.001,0.001]
    
    im_size = 96
    model_name = 'khorrami_ck_96'
    # model_name = 'khorrami_ck_96_caps_bl'
    save_after = 10
    # margin_params = {'step':1,'start':0.2}
    # strs_append = '_'.join([str(val) for val in [model_name,300]+dec_after+lr])
    # out_dir_train = os.path.join(out_dir_meta,'ck_'+str(split_num)+'_'+strs_append)
    # model_file = os.path.join(out_dir_train,'model_299.pt')
    model_file=None    
    if not mmi:
        strs_append = '_'.join([str(val) for val in ['train_test_files_non_peak_one_third',model_name,num_epochs]+dec_after+lr])
        strs_append = '_'+strs_append
        pre_pend = 'ck_'
        folds = range(10)
    else:
        pre_pend = 'mmi_96_'
        folds = range(2)
        strs_append = '_'.join([str(val) for val in ['train_test_files',model_name,num_epochs]+dec_after+lr])
        strs_append = '_'+strs_append
        

    if model_to_test is None:
        model_to_test= num_epochs -1 

    for split_num in folds:
        out_dir_train = os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
        print out_dir_train



        out_file_model = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
        if os.path.exists(out_file_model):
            print 'skipping',out_file_model
            # continue
        else:
            print 'not done',out_file_model
            raw_input()

        if not mmi:
            train_file = '../data/ck_96/train_test_files_non_peak_one_third/train_'+str(split_num)+'.txt'
            test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
            test_file_easy = '../data/ck_96/train_test_files_non_peak_one_third/test_'+str(split_num)+'.txt'
            mean_file = '../data/ck_96/train_test_files_non_peak_one_third/train_'+str(split_num)+'_mean.png'
            std_file = '../data/ck_96/train_test_files_non_peak_one_third/train_'+str(split_num)+'_std.png'
        else:
            type_data = 'train_test_files'; n_classes = 6;
            train_pre = os.path.join('../data/mmi',type_data)
            test_pre = train_pre
            train_file = os.path.join(train_pre,'train_'+str(split_num)+'.txt')
            test_file_easy = os.path.join(train_pre,'test_front_'+str(split_num)+'.txt')        
            test_file = os.path.join(test_pre,'test_side_'+str(split_num)+'.txt')
            mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
            std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')
        

        # train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
        # test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
        # mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
        # std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'
        
        mean_im = scipy.misc.imread(mean_file).astype(np.float32)
        std_im = scipy.misc.imread(std_file).astype(np.float32)
        std_im[std_im==0]=1.

        if not mmi:
            list_of_to_dos = ['pixel_augment','flip','rotate','scale_translate']
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
            test_data_easy = dataset.CK_96_Dataset(test_file_easy, mean_file, std_file, data_transforms['val'])
        else:
            list_of_to_dos = ['flip','rotate','scale_translate']
            data_transforms = {}
            data_transforms['train']= transforms.Compose([
                lambda x: augmenters.random_crop(x,im_size),
                lambda x: augmenters.augment_image(x,list_of_to_dos),
                transforms.ToTensor(),
                lambda x: x*255.
            ])
            data_transforms['val']= transforms.Compose([
                transforms.ToTensor(),
                lambda x: x*255.
                ])

            print train_file
            print test_file
            print std_file
            print mean_file
            # raw_input()

            train_data = dataset.CK_96_Dataset_with_rs(train_file, mean_file, std_file, data_transforms['train'])
            test_data_easy = dataset.CK_96_Dataset_with_rs(test_file_easy, mean_file, std_file, data_transforms['val'],resize = im_size)
            test_data = dataset.CK_96_Dataset_with_rs(test_file, mean_file, std_file, data_transforms['val'],resize = im_size)
            



        
        network_params = dict(n_classes=8,bn=False)
        
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
                    criterion = nn.CrossEntropyLoss(),
                    gpu_id = 1,
                    num_workers = 0,
                    model_file = model_file,
                    epoch_start = epoch_start,
                    network_params = network_params)

        test_params =  dict(out_dir_train = out_dir_train,
                            model_num = model_to_test, 
                            train_data = train_data,
                            test_data = test_data, 
                            gpu_id = 1,
                            model_name = model_name,
                            batch_size_val = batch_size_val,
                            criterion = nn.CrossEntropyLoss(),
                            margin_params  = None,
                            network_params = network_params,
                            post_pend = '',
                            model_nums = None)

        test_params_easy =  dict(out_dir_train = out_dir_train,
                            model_num = model_to_test, 
                            train_data = train_data,
                            test_data = test_data_easy, 
                            gpu_id = 1,
                            model_name = model_name,
                            batch_size_val = batch_size_val,
                            criterion = nn.CrossEntropyLoss(),
                            margin_params  = None,
                            network_params = network_params,
                            post_pend = '_easy',
                            model_nums = None)

        print train_params
        param_file = os.path.join(out_dir_train,'params.txt')
        all_lines = []
        for k in train_params.keys():
            str_print = '%s: %s' % (k,train_params[k])
            print str_print
            all_lines.append(str_print)
        util.writeFile(param_file,all_lines)

        # train_model(**train_params)
        test_model(**test_params)
        # print test_params['test_data']
        # print test_params['post_pend']
        # # raw_input()
        # print test_params_easy['test_data']
        # print test_params_easy['post_pend']
        
        test_model(**test_params_easy)
        # print out_dir_train, model_to_test
        # raw_input()

    getting_accuracy.print_accuracy(out_dir_meta,pre_pend,strs_append,folds,log='log.txt')
    getting_accuracy.view_loss_curves(out_dir_meta,pre_pend,strs_append,folds,num_epochs-1)



def main():
    for model_to_test in [299]:
    # range(0,300,10):
        khorrami_bl_exp(mmi=True,model_to_test = model_to_test)

    # for model_to_test in range(0,300,50)+[299]:
    #     khorrami_bl_exp(model_to_test)

if __name__=='__main__':
    main()