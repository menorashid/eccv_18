from helpers import util, visualize
import random
import scipy.misc
from PIL import Image

import torch.utils
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms
import models
import matplotlib.pyplot as plt
import time
import os
import h5py
import itertools
import glob
import sklearn.metrics

import os
from helpers import util,visualize,augmenters
import random
import dataset
import numpy as np
from models.spread_loss import Spread_Loss

import torch.nn.functional as F

dir_server = '/disk3'
str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
click_str = 'http://vision3.idav.ucdavis.edu:1000'

def save_recon_variants(out_dir_train,
                model_num,
                train_data,
                test_data,
                gpu_id = 0,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss(),
                margin_params = None,
                network_params = None,barebones = True):

    # out_dir_train = out_dir_train,
    #                     model_num = model_num_curr, 
    #                     train_data = train_data,
    #                     test_data = test_data,
    #                     gpu_id = 0,
    #                     model_name = model_name,
    #                     batch_size_val = batch_size_val,
    #                     criterion = criterion,
    #                     margin_params = margin_params,
    #                     network_params = network_params,barebones=False

    mag_range = np.arange(-0.5,0.6,0.1)
    out_dir_results = os.path.join(out_dir_train,'vary_a_batch_squash_'+str(model_num))
    print out_dir_results
    util.mkdir(out_dir_results)
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    log_arr=[]

    # network = models.get(model_name,network_params)
    
    if batch_size_val is None:
        batch_size_val = len(test_data)
    

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_val,
                        shuffle=False, num_workers=1)

    torch.cuda.device(0)
    iter_begin = 0
    model = torch.load(model_file)
    model.cuda()
    model.eval()
    
    predictions = []
    labels_all = []
    out_all = []
    caps_all = []
    recons_all = {}

    mean_im = test_data.mean
    
    mean_im = mean_im[np.newaxis,:,:]
    std_im = test_data.std[np.newaxis,:,:]


    for num_iter,batch in enumerate(test_dataloader):
            
        # batch = test_dataloader.next() 
        if criterion=='marginmulti':
            labels = Variable(batch['label'].float().cuda())
        else:
            labels = Variable(torch.LongTensor(batch['label']).cuda())
        # labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())

        recons_all[(0,0)] = data.data.cpu().numpy()
        # labels = Variable(torch.LongTensor(batch['label']).cuda())
        

        # output, caps = 
        all_out = model(data, return_caps = True)
        caps = all_out[-1]    

        
        # print caps
        classes = all_out[0]
        pred = F.relu(classes - 0.5)
        pred = torch.ceil(pred) 
        print pred


        caps_unit = caps/classes.view(classes.size(0),classes.size(1),1)
        # classes_unit = (caps_unit ** 2).sum(dim=-1) ** 0.5        

        labels_made_up = np.zeros((labels.size(0),labels.size(1)))
        for label_on in range(labels_made_up.shape[1]):
            labels_made_up = labels_made_up*0
            labels_made_up[:,label_on] = 1
            labels_curr = Variable(torch.Tensor(labels_made_up).float().cuda())
            for attr_num in range(caps.size(2)):
                for mag_curr in mag_range:
                # np.arange(-0.25,0.25,0.05):
                    caps_curr = caps_unit.clone()
                    caps_curr[:,:,attr_num]=mag_curr
                    # caps_mag = (caps_curr ** 2).sum(dim=-1) ** 0.5
                    # caps_curr = caps_curr/caps_mag.view(caps_mag.size(0),caps_mag.size(1),1)
                    recon_curr = model.just_reconstruct(caps_curr,labels_curr)
                    recons_all[(label_on,attr_num,mag_curr)] = recon_curr.data.cpu().numpy()


            # for mag in np.arange(0.1,1.1,0.1):
            #     print mag
            #     caps_curr = torch.mul(caps_unit,mag)
            #     print caps_curr.size()
            #     recon_max = model.just_reconstruct(caps_curr,labels)
            #     recons_all[(0,mag)] = recon_max.data.cpu().numpy()


        # print classes_unit
        # print classes.shape
        # print caps.shape

        print labels
        labels_keep = np.logical_and(labels.data.cpu().numpy(), pred.data.cpu().numpy())
        print labels_keep

        break



    #     recons_all[(-1,-2)] = all_out[1].data.cpu().numpy()
    #     recons_all[(-1,-1)] = model.just_reconstruct(caps,labels).data.cpu().numpy()

    #     print caps.size()
    #     caps_data = caps.data.cpu().numpy()

    #     for dim_num in range(caps.size(2)):
            
            

    #         for inc_curr in np.arange(-0.25,0.30,0.05):
    #             caps = torch.autograd.Variable(torch.Tensor(caps_data)).cuda()
    #             caps[:,:,dim_num]=inc_curr
    #             squared_norm = (caps ** 2).sum(dim=2, keepdim=True)
    #             scale = squared_norm / (1 + squared_norm)
    #             caps = scale * caps / torch.sqrt(squared_norm)

    #             recons_curr = model.just_reconstruct(caps,labels)
    #             recons_all[(dim_num,inc_curr)]=recons_curr.data.cpu().numpy()
    #     # recons_curr = 
    #     break

    ims_html = []
    captions_html = []
    out_file_html = os.path.join(out_dir_results,'rel_im_mag_only.html')
    out_dir_results = out_dir_results.replace(str_replace[0],str_replace[1])
    
    num_im = labels_keep.shape[0]
    num_labels = labels_keep.shape[1]
    num_attr = 32
    for im_num in range(num_im):
        # key_curr in recons_all.keys():
        # label_on, attr_num, mag_curr = key_curr
        print im_num, labels_keep[im_num],np.where(labels_keep[im_num])
        # raw_input()
        for label_num in np.where(labels_keep[im_num])[0]:
            print im_num, label_num
            for attr_num in range(32):
                im_row = []
                caption_row = []
                for mag_curr in mag_range:
                    key_curr = (label_num,attr_num,mag_curr)
                    im_curr = recons_all[key_curr][im_num]

                    out_file = '_'.join([str(val) for val in [im_num,label_on,attr_num,mag_curr,'.jpg']])
                    out_dir_curr = os.path.join(out_dir_results,'label_'+str(label_on))
                    util.mkdir(out_dir_curr)

                    out_file = os.path.join(out_dir_curr,out_file)
                    im_curr = im_curr*std_im+mean_im
                    scipy.misc.imsave(out_file,im_curr[0])

                    im_row.append('./'+util.getRelPath(out_file,dir_server))
                    caption_str = '%d %d %d %.2f' % tuple([im_num]+list(key_curr))
                    caption_row.append(caption_str)
                        # ' '.join([str(val) for val in [im_num,key_str]]))

                ims_html.append(im_row)
                captions_html.append(caption_row)

    visualize.writeHTML(out_file_html,ims_html,captions_html,96,96)
    print out_file_html.replace(dir_server,click_str)




        # recons = recons_all[key_curr]
        # recons = (recons*std_im)+mean_im
        # # out_dir = os.path.join(out_dir_results,'%d_%.2f'%(key_curr[0],key_curr[1]))
        # # util.mkdir(out_dir)
        # key_pre = '%d_%.2f'%(key_curr[0],key_curr[1])
        # for idx_im_curr,im_curr in enumerate(recons):
        #     scipy.misc.imsave(os.path.join(out_dir_results,key_pre+'_'+str(idx_im_curr)+'.jpg'),im_curr[0])
        # visualize.writeHTMLForFolder(out_dir_results,'.jpg')

