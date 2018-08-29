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
# from helpers import util,visualize,augmenters
import save_visualizations
from torch.autograd import Variable

def train_vgg(wdecay, lr, folds=[4,9], model_name='vgg_capsule_bp4d', epoch_stuff=[30,60],res=False, class_weights = False, exp = False, align = False, disfa = False,more_aug=False, model_to_test = None, gpu_id = 0, save_after = 1):
    out_dirs = []

    out_dir_meta = '../experiments/'+model_name
    num_epochs = epoch_stuff[1]

    if model_to_test is None:
        model_to_test = num_epochs-1

    epoch_start = 0
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr

    im_resize = 256
    im_size = 224
    if not disfa:
        dir_files = '../data/bp4d'
        if align:
            type_data = 'train_test_files_256_color_align'; n_classes = 12;
        else:
            type_data = 'train_test_files_256_color_nodetect'; n_classes = 12;
        pre_pend = 'bp4d_256_'+type_data+'_'
        binarize = False
    else:
        dir_files = '../data/disfa'
        type_data = 'train_test_8_au_all_method_256_color_align'; n_classes = 8;
        pre_pend = 'disfa_'+type_data+'_'
        binarize = True
        pre_pend = 'disfa_256_'+type_data+'_'
    
    criterion_str = 'MultiLabelSoftMarginLoss'
    criterion = nn.MultiLabelSoftMarginLoss()
    # nn.MultiMarginLoss()
    # torch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=None, reduce=None, reduction='elementwise_mean')

    init = False

    strs_append_list = [class_weights,criterion_str,num_epochs]+dec_after+lr+[more_aug]
    strs_append = '_'+'_'.join([str(val) for val in strs_append_list])
    
    
    
    for split_num in folds:
        
        
        out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
        final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
        # final_model_file = os.path.join(out_dir_train,'results_model_'+str(model_to_test))
        if os.path.exists(final_model_file):
            print 'skipping',final_model_file
            # continue 
        else:
            print 'not skipping', final_model_file
            
        train_file = os.path.join(dir_files,type_data,'train_'+str(split_num)+'.txt')
        test_file = os.path.join(dir_files,type_data,'test_'+str(split_num)+'.txt')

        if 'imagenet' in model_name:
            bgr= False
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            std_div = None

            data_transforms = {}
            data_transforms['train'] = [ transforms.ToPILImage(),
                transforms.RandomCrop(im_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                normalize]

            data_transforms['val'] = [transforms.ToPILImage(),
                transforms.Resize((im_size,im_size)),
                transforms.ToTensor(),
                normalize]
            
            if torch.version.cuda.startswith('9'):
                data_transforms['train'].append(lambda x: x.float())
                data_transforms['val'].append(lambda x: x.float())

            data_transforms['train']= transforms.Compose(data_transforms['train'])
            data_transforms['val']= transforms.Compose(data_transforms['val'])

            train_data = dataset.Bp4d_Dataset(train_file, bgr = bgr, binarize = binarize, transform = data_transforms['train'])
            test_data = dataset.Bp4d_Dataset(test_file, bgr = bgr, binarize= binarize, transform = data_transforms['val'])
        else:
            bgr= True
            mean_std = np.array([[93.5940,104.7624,129.1863],[1.,1.,1.]]) #bgr
            
            normalize = transforms.Normalize(mean=mean_std[0,:],
                                     std=mean_std[1,:])
            
            data_transforms = {}
            data_transforms['train'] = [ transforms.ToPILImage(),
                transforms.RandomCrop(im_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                lambda x: x*255,
                normalize]

            data_transforms['val'] = [transforms.ToPILImage(),
                transforms.Resize((im_size,im_size)),
                transforms.ToTensor(),
                lambda x: x*255,
                normalize]
            
            if torch.version.cuda.startswith('9'):
                data_transforms['train'].append(lambda x: x.float())
                data_transforms['val'].append(lambda x: x.float())

            data_transforms['train']= transforms.Compose(data_transforms['train'])
            data_transforms['val']= transforms.Compose(data_transforms['val'])

            train_data = dataset.Bp4d_Dataset(train_file, bgr = bgr, binarize = binarize, transform = data_transforms['train'])
            test_data = dataset.Bp4d_Dataset(test_file, bgr = bgr, binarize= binarize, transform = data_transforms['val'])


        class_weights = util.get_class_weights_au(util.readLinesFromFile(train_file))
        criterion._buffers['weight'] = torch.Tensor(class_weights)
        

        network_params = dict(n_classes=n_classes, to_init = ['last_fc'])
            
        batch_size = 32
        batch_size_val = 32
        
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
                    gpu_id = gpu_id,
                    num_workers = 0,
                    epoch_start = epoch_start,
                    margin_params = None,
                    network_params = network_params,
                    weight_decay=wdecay)
        
        test_params = dict(out_dir_train = out_dir_train,
                    model_num = model_to_test, 
                    train_data = train_data,
                    test_data = test_data,
                    gpu_id = gpu_id,
                    model_name = model_name,
                    batch_size_val = batch_size_val,
                    criterion = criterion,
                    margin_params = None,
                    network_params = network_params,
                    barebones=True)
        
        print train_params
        param_file = os.path.join(out_dir_train,'params.txt')
        all_lines = []
        for k in train_params.keys():
            str_print = '%s: %s' % (k,train_params[k])
            print str_print
            all_lines.append(str_print)
        
        train_model(**train_params)

        test_model(**test_params)
        
        
    getting_accuracy.print_accuracy(out_dir_meta,pre_pend,strs_append,folds,log='log.txt')


def imagenet_experiments():
    wdecay = 0
    route_iter = 3
    folds =[1,2]
    model_name =  'vgg_imagenet_finetune' 
    reconstruct = True
    loss_weights = [1.,0.1]
    
    # epoch_stuff = [350,5]
    # exp = True

    epoch_stuff = [10,10]
    exp = False
    align = True
    lr = [0,0.001,0.001,0.001]
    save_after = 1
    train_vgg(wdecay= wdecay,
        lr = lr,
        folds=folds,
        model_name=model_name,
        epoch_stuff=epoch_stuff,
        exp = exp ,
        align = align ,
        save_after = save_after)

def main():
    imagenet_experiments()
    




if __name__=='__main__':
    main()