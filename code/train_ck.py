import os
from helpers import util,visualize
import random
from train_test import train_model, test_model
import torch.nn as nn
import dataset
from torchvision import transforms
import numpy as np
import scipy.misc

def under_bed():
    split_num =0
    train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
    test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
    mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
    std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'

    train_file_curr = util.readLinesFromFile(train_file)[0]
    mean = scipy.misc.imread(mean_file).astype(np.float32)
    std = scipy.misc.imread(std_file).astype(np.float32)
    std[std==0]=1.
    train_file_curr,label = train_file_curr.split(' ')
    label = int(label)
    image = scipy.misc.imread(train_file_curr).astype(np.float32)
    # image = (image-mean)/std
    # image = image[:,:,np.newaxis]
    
    out_dir = '../scratch/check_ck_aug'
    util.mkdir(out_dir)

    out_file_bef = os.path.join(out_dir,'im.jpg')
    scipy.misc.imsave(out_file_bef,image)
    list_of_to_dos = ['pixel_augment']
    out_file_aft = os.path.join(out_dir,'im_'+'_'.join(list_of_to_dos)+'.jpg')

    import torch
    data_transforms = {}
    data_transforms['train']= transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.5),
                lambda x: augment_image(x,list_of_to_dos,mean_im = mean,std_im = std),
                transforms.ToTensor(),
            ])
    
    train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file,data_transforms['train'])
    train_dataloader = torch.utils.data.DataLoader(train_data, 
                        batch_size=1,
                        shuffle=False, 
                        num_workers=0)

    for batch in train_dataloader:
        print batch.keys()
        print torch.min(batch['image']),torch.max(batch['image'])
        print  batch['label'].shape
        image = batch['image'][0].numpy()
        print image.shape
        break
    
    scipy.misc.imsave(out_file_aft,image[0])
    visualize.writeHTMLForFolder(out_dir)

def augment_image( im, list_of_to_dos = ['flip','rotate','scale_translate'],mean_im=None, std_im=None, im_size = 96):
        # khorrami augmentation for ck+
        # trying to get baseline results 

        # a. Flip: The image is horizontally mirrored with probability 0.5.
        # b. Rotation: A random theta is sampled uniformly from the range [-5, 5] degrees and the image is rotated by theta.
        # c. Scale: A random alpha is sampled uniformly from the range [0.7, 1.4] and the image is scaled by alpha.
        # d. Translation: A random [x, y] vector is sampled and the image is translated by [x, y]. x and y are defined such that:
        # x ~ Uniform(-delta/2, delta/2)
        # y ~ Uniform(-delta/2, delta/2)
        # where delta = (alpha-1)*96.
        # e. Intensity Change: The pixels of an image (p(i, j)) are changed using the following formula: 
        # p*(i, j) = (p(i, j)^a) * b + c 
        # where a, b, and c are defined as:
        # a ~ Uniform(0.25, 4)
        # b ~ Uniform(0.7, 1.4)
        # c ~ Uniform(-0.1, 0.1)
        
        rot_range =[-5,5]
        alpha_range = [0.7,1.4]
        a_range = [0.25,4]
        b_range = [0.7, 1.4]
        # b_range = [0.5,1]
        c_range = [-0.1,0.1]
        # im_size = 96
        # print 'bef',np.min(im), np.max(im)
        # print 'bef',np.min(im), np.max(im)
        
        if 'pixel_augment' in list_of_to_dos:
            im = im[:,:,0]
            # print np.min(im),np.max(im)
            # print np.min(std_im),np.max(std_im)
            # print np.min(mean_im),np.max(mean_im)
            im = im*std_im
                # )+mean_im
            # print 'aft normal',np.min(im),np.max(im)
            im = np.clip(im + mean_im,0,255)
                # )+mean_im
            # print 'aft normal',np.min(im),np.max(im)
            
            im = im/255.
            

            a_b_c = np.random.random_sample((3,))
            a = a_b_c[0]*(a_range[1]-a_range[0]) + a_range[0]
            b = a_b_c[1]*(b_range[1]-b_range[0]) + b_range[0]
            c = a_b_c[2]*(c_range[1]-c_range[0]) + c_range[0]
            # print a,b,c
            # min_max = [np.min(im),np.max(im)]
            # im = (im - min_max[0])/float(min_max[1] - min_max[0])
            # print np.min(im),np.max(im)
            im = (im**a)*b + c
            # print np.min(im),np.max(im)
            # im = (im - np.min(im))/float(np.max(im)-np.min(im))
            # print np.min(im),np.max(im)
            # im = (im * float(min_max[1] - min_max[0])) + min_max[0]
            im = im*255.
            im = (im - mean_im)/std_im
            im = im[:,:,np.newaxis]
            # print np.min(im),np.max(im)
            # raw_input()

        im = np.concatenate((im,im,im),2)
        min_im = np.min(im)
        im = im-min_im
        max_im = np.max(im)
        im = im/max_im 

        

        # flip it
        if 'flip' in list_of_to_dos:
            if np.random.random()<0.5:
                im = im[:,::-1,:]

        if 'rotate' in list_of_to_dos:
            deg = np.random.random()*(rot_range[1]-rot_range[0]) + rot_range[0]
            im = scipy.misc.imrotate(im,deg)

        if 'scale_translate' in list_of_to_dos:
            alpha = np.random.random()*(alpha_range[1]-alpha_range[0]) + alpha_range[0]
            delta = abs(alpha-1)*im_size
            delta_range = [-1*delta/2.,delta/2.]
            assert delta_range[0]<=delta_range[1]

            im_rs = scipy.misc.imresize(im, alpha)
            
            vec_translate = np.random.random_sample((2,))*(delta_range[1]-delta_range[0]) + delta_range[0]
            vec_translate = np.around(vec_translate).astype(dtype = np.int)
            padding = [0,0,0,0]
            for dim_num in range(2):
                if vec_translate[dim_num]<0:
                    padding[2+dim_num] = -vec_translate[dim_num]
                else:
                    padding[dim_num] = vec_translate[dim_num]

            im_rs = np.pad(im_rs,((padding[0],padding[1]),(padding[2],padding[3]),(0,0)),'constant')
            
            start_idx_im = [0,0]
            end_idx_im = [0,0]
            start_idx_im_rs = [0,0]
            end_idx_im_rs = [0,0]

            for dim_num in range(2):
                if im_rs.shape[dim_num]<im_size:
                    start_idx_im_rs[dim_num] = 0
                    end_idx_im_rs[dim_num] = im_rs.shape[dim_num]
                    start_idx_im[dim_num] = max(int(round(im_size/float(2) -im_rs.shape[dim_num]/float(2))),0)
                else:
                    start_idx_im_rs[dim_num] = max(int(round(im_rs.shape[dim_num]/float(2) - im_size/float(2) )),0)
                    end_idx_im_rs[dim_num] = min(start_idx_im_rs[dim_num]+im_size,im_rs.shape[dim_num])

                    start_idx_im[dim_num] = 0
                end_idx_im[dim_num] = min(start_idx_im[dim_num] + (end_idx_im_rs[dim_num]-start_idx_im_rs[dim_num]),im_size)


            im = np.zeros(im.shape)
            
            im[start_idx_im[0]:end_idx_im[0],start_idx_im[1]:end_idx_im[1],:] = im_rs[start_idx_im_rs[0]:end_idx_im_rs[0],start_idx_im_rs[1]:end_idx_im_rs[1],:]

        im = im[:,:,:1]
        if 'rotate' in list_of_to_dos or 'scale_translate' in list_of_to_dos:
            im = im/255.

        im = im * max_im
        im = im + min_im
        # print 'aft',np.min(im), np.max(im)
        return im

def main():

    # out_dir_meta = '../experiments/bl_khorrami_ck_96_nobn_pixel_augment_255_range'
    # range_splits = [0,1,2,3,4,5]
    out_dir_meta = '../experiments/bl_khorrami_ck_96_nobn_pixel_augment_255_range_trans_fix'
    range_splits = range(6,10)
    print range_splits
    
    # range(10)
    
    util.mkdir(out_dir_meta)
    all_accuracy = []

    for split_num in range_splits:
        
        train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
        test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
        mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
        std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'

        list_of_to_dos = ['flip','rotate','scale_translate','pixel_augment']
        mean_im = scipy.misc.imread(mean_file).astype(np.float32)
        std_im = scipy.misc.imread(std_file).astype(np.float32)
    
        batch_size = 128
        batch_size_val = None
        num_epochs = 500
        save_after = 100
        disp_after = 1
        plot_after = 10
        test_after = 1
        
        lr = [0.001, 0.001]
        # lr = [0.0001,0.0001]
        dec_after = 300 
        model_name = 'khorrami_ck_96'
        criterion = nn.CrossEntropyLoss()
        gpu_id = 0
        num_workers = 2
        model_num = num_epochs-1

        # model_file = None
        # epoch_start = 0
        # lr_dir_train = lr
        
        lr_dir_train = [0.01, 0.01]
        # strs_append = '_'.join([str(val) for val in [num_epochs,dec_after,lr_dir_train[0],lr_dir_train[1],'100_dec']])
        strs_append = '_'.join([str(val) for val in [num_epochs,dec_after,lr_dir_train[0],lr_dir_train[1]]])
        out_dir_train = os.path.join(out_dir_meta,'split_'+str(split_num)+'_'+strs_append)
        print out_dir_train
        
        
        epoch_start = 401
        strs_append = '_'.join([str(val) for val in [400,300,lr_dir_train[0],lr_dir_train[1]]])
        out_dir_res = os.path.join(out_dir_meta,'split_'+str(split_num)+'_'+strs_append)
        # strs_append = '_'.join([str(val) for val in [250,200,lr_dir_train[0],lr_dir_train[1]]])
        # model_file = os.path.join(out_dir_meta,'split_'+str(split_num)+'_'+strs_append,'model_200.pt')
        model_file = os.path.join(out_dir_res,'model_399.pt')
        
        
        

        # raw_input()
        util.mkdir(out_dir_train);

        data_transforms = {}
        data_transforms['train']= transforms.Compose([
            lambda x: augment_image(x,list_of_to_dos,mean_im = mean_im,std_im = std_im),
            transforms.ToTensor(),
            lambda x: x*255.
        ])
        data_transforms['val']= transforms.Compose([
            transforms.ToTensor(),
            lambda x: x*255.
            ])

        train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
        test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
        
        train_model(out_dir_train,
                    train_data,
                    test_data,
                    batch_size = batch_size,
                    batch_size_val = batch_size_val,
                    num_epochs = num_epochs,
                    save_after = save_after,
                    disp_after = disp_after,
                    plot_after = plot_after,
                    test_after = test_after,
                    lr = lr,
                    dec_after = dec_after,
                    model_name = model_name,
                    criterion = criterion,
                    gpu_id = gpu_id,
                    num_workers = num_workers,
                    model_file = model_file,
                    epoch_start = epoch_start)

        test_model(out_dir_train,
                    model_num,
                    train_data,
                    test_data,
                    model_name = model_name,
                    batch_size_val = batch_size_val,
                    criterion = criterion)
        res_dir = os.path.join(out_dir_train,'results_model_'+str(model_num))
        log_file = os.path.join(res_dir,'log.txt')
        accuracy = util.readLinesFromFile(log_file)[-1]
        accuracy = float(accuracy.split(' ')[1])
        all_accuracy.append(accuracy)

    print all_accuracy,np.mean(all_accuracy),np.std(all_accuracy)


if __name__=='__main__':
    main()