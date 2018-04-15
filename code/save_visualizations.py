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


def save_all_im(out_dir_im,pre_vals,im_out,post_pend):
    # print pre_vals
    pre_im = os.path.join(out_dir_im,'_'.join([str(val) for val in pre_vals]))
    # print pre_im
    ims_row =[]
    for im_out_curr,post_pend_curr in zip(im_out,post_pend):
        out_file = pre_im+'_'+'_'.join([str(val) for val in post_pend_curr])+'.jpg'
        # print out_file
        # print im_out_curr.shape
        scipy.misc.imsave(out_file,im_out_curr.squeeze())
        ims_row.append(out_file)
    return ims_row
    

def save_primary_caps(out_dir_train,
                model_num,
                train_data,
                test_data,
                gpu_id = 0,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss(),
                margin_params = None,
                network_params = None,
                barebones = True,
                au=False,
                class_rel = 0
                ):
    
    mag_range = np.arange(-0.5,0.6,0.1)
    out_dir_results = os.path.join(out_dir_train,'save_primary_caps_train_data_'+str(model_num))
    util.makedirs(out_dir_results)
    out_dir_im = os.path.join(out_dir_results,'im_save')
    util.mkdir(out_dir_im)

    print out_dir_results
    
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
    
    # predictions = []
    # labels_all = []
    # out_all = []
    # caps_all = []
    # recons_all = {}

    mean_im = test_data.mean
    
    mean_im = mean_im[np.newaxis,:,:]
    std_im = test_data.std[np.newaxis,:,:]

    for num_iter,batch in enumerate(test_dataloader):
        print 'NUM ITER',num_iter

        # batch = test_dataloader.next() 
        if criterion=='marginmulti':
            labels = Variable(batch['label'].float().cuda())
        else:
            labels = Variable(torch.LongTensor(batch['label']).cuda())
        # labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())

        # recons_all[(0,0)] = data.data.cpu().numpy()
        # labels = Variable(torch.LongTensor(batch['label']).cuda())
        
        x = model.features(data)
        # print model
        _,routes = model.caps.forward_intrusive(x)
        x = x.data.cpu().numpy()
        
        # print x.shape
        # print routes[1].shape
        out_file_curr = os.path.join(out_dir_results,str(num_iter)+'.npy')
        out_file_routes = os.path.join(out_dir_results,str(num_iter)+'_routes.npy')
        print out_file_curr,out_file_routes
        
        np.save(out_file_curr,x)
        np.save(out_file_routes,routes[1])
        # print x.size()
            
        # break



def save_routings(out_dir_train,
                model_num,
                train_data,
                test_data,
                gpu_id = 0,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss(),
                margin_params = None,
                network_params = None,
                barebones = True,
                au=False
                ):
    
    mag_range = np.arange(-0.5,0.6,0.1)
    out_dir_results = os.path.join(out_dir_train,'save_routings_single_batch_'+str(model_num))
    util.makedirs(out_dir_results)
    out_dir_im = os.path.join(out_dir_results,'im_save')
    util.mkdir(out_dir_im)

    print out_dir_results
    
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
    
    # predictions = []
    # labels_all = []
    # out_all = []
    # caps_all = []
    # recons_all = {}

    mean_im = test_data.mean
    
    mean_im = mean_im[np.newaxis,:,:]
    std_im = test_data.std[np.newaxis,:,:]


    for num_iter,batch in enumerate(test_dataloader):
        print 'NUM ITER',num_iter

        # batch = test_dataloader.next() 
        if criterion=='marginmulti':
            labels = Variable(batch['label'].float().cuda())
        else:
            labels = Variable(torch.LongTensor(batch['label']).cuda())
        # labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())

        # recons_all[(0,0)] = data.data.cpu().numpy()
        # labels = Variable(torch.LongTensor(batch['label']).cuda())
        
        if not au:
            x = model.features(data)
            _,routes = model.caps.forward_intrusive(x)
        else:
            x = model.vgg_base(data)
            x = model.features[0](x)
            _,routes = model.features[1].forward_intrusive(x)
        
        routes = [np.squeeze(routes_curr) for routes_curr in routes]
        print len(routes), routes[0].shape
        # raw_input()

        # output, caps = 
        classes,reconstructions_gt,_,caps = model(data, labels, return_caps = True)
        classes, reconstructions_gt, caps = [val.data.cpu().numpy() for val in [classes, reconstructions_gt,caps]]
        labels = labels.data.cpu().numpy()
        

        if not au:
            preds = np.argmax(classes,1)
        else:
            preds = classes
            preds[preds<=0.5]=0
            preds[preds>0.5]=1

        print preds.shape, labels.shape
        print np.sum(preds==labels)/float(labels.size)
        

        batch_size = data.shape[0]

        data = data.data.cpu().numpy()


        ims_all = []
        for im_num in range(batch_size):
            post_pend = []
            im_out = []

            gt_label = labels[im_num]
            pred_label = preds[im_num]
            # print 'gt',gt_label,'pred',pred_label
            im_in = data[im_num]
            # [0]
            # print im_in.shape
            # raw_input()
            im_in = (im_in*std_im)+mean_im
            # print im_in.shape
            im_out.append(im_in)
            post_pend.append(['org'])

            recon_gt = reconstructions_gt[im_num]
            # [0]
            recon_gt = (recon_gt*std_im)+mean_im
            # print recon_gt.shape
            im_out.append(recon_gt)
            post_pend.append(['recon_gt'])
            routes_im = [np.sum(route[:,im_num,:,:],2) for route in routes]
            # for val in im_out:
            #     print val.shape
            
            for label_curr in range(routes_im[0].shape[0]):
                for route_idx, route_curr in enumerate(routes_im):
                    route_curr = np.array(route_curr)
                    route_curr = np.reshape(route_curr,(route_curr.shape[0],32,6,6))
                    
                    # route_curr = np.reshape(route_curr,(route_curr.shape[0],6,6,32))
                    # # print route_curr.shape
                    # route_curr = np.transpose(route_curr,(0,3,1,2))
                    # # print route_curr.shape
                    # raw_input()
                    
                    # print np.min(route_curr), np.max(route_curr)
                    route_curr = np.sum(route_curr,1)
                    # raw_input()
                    # print route_curr.shape
                    # route_curr = route_curr-np.min(route_curr)
                    # route_curr = route_curr/np.max(route_curr)
                    # print route_curr.shape

                
                    route_label = route_curr[label_curr]
                    # route_label =np.reshape(route_curr[label_curr],(32,6,6))
                    # route_label = np.sum(route_label,0)
                    im_out.append((route_label))
                    post_pend.append([label_curr,route_idx])

                    # print label_curr, route_label.shape, 
                    # print np.min(route_label),np.max(route_label)

            pre_vals = [im_num,gt_label,pred_label]
            ims_row = save_all_im(out_dir_im,pre_vals,im_out,post_pend)
            ims_all.append(ims_row)
            # print ims_all
            # print len(ims_all)
            # print len(ims_all[0])
            # raw_input()
        # break
    
    # mats_to_save = []
        mats_to_save = [labels,preds,routes[0],routes[1]]
        mats_names = ['labels_'+str(num_iter),'preds_'+str(num_iter),'routes_0_'+str(num_iter),'routes_1_'+str(num_iter)]
        for mat_curr, file_curr in zip(mats_to_save,mats_names):
            out_file_curr = os.path.join(out_dir_results,file_curr+'.npy')
            np.save(out_file_curr,mat_curr)

        np.save(os.path.join(out_dir_results,'ims_all_'+str(num_iter)+'.npy'),np.array(ims_all))
    return num_iter


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


def save_class_vary_mag(out_dir_train,
                model_num,
                train_data,
                test_data,
                gpu_id = 0,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss(),
                margin_params = None,
                network_params = None,
                barebones = True,
                class_rel = 0,
                au = False
                ):
    
    mag_range = np.arange(0.1,1.0,0.1)
    out_dir_results = os.path.join(out_dir_train,'save_class_vary_mag_single_batch_'+str(model_num))
    util.makedirs(out_dir_results)
    out_dir_im = os.path.join(out_dir_results,'im_save')
    util.mkdir(out_dir_im)

    print out_dir_results
    
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

    ims_all = []
    for num_iter,batch in enumerate(test_dataloader):
        print 'NUM ITER', num_iter
        
        # labels = batch['label'].cpu().numpy()
        # data = batch['image'].cpu().numpy()
        # rel_vals = labels==class_rel
        # labels = labels[rel_vals]
        # data = data[rel_vals]
        # batch['image'] = torch.Tensor(data)
        # batch['label'] = torch.LongTensor(labels)
        # print labels.shape
        # print data.shape

        # raw_input()

        # batch = test_dataloader.next() 
        if criterion=='marginmulti':
            labels = Variable(batch['label'].float().cuda())
        else:
            labels = Variable(torch.LongTensor(batch['label']).cuda())
        # labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())

        # recons_all[(0,0)] = data.data.cpu().numpy()
        # labels = Variable(torch.LongTensor(batch['label']).cuda())
        
        # x = model.features(data)
        # _,routes = model.caps.forward_intrusive(x)
        # routes = [np.squeeze(routes_curr) for routes_curr in routes]
        # print len(routes), routes[0].shape


        # output, caps = 
        classes,reconstructions_gt,_,caps = model(data, labels, return_caps = True)
        
        caps_mag = (caps ** 2).sum(dim=-1) ** 0.5
        caps_unit = caps/caps_mag.view(caps_mag.size(0),caps_mag.size(1),1)

        recons_all = []
        for mag_curr in mag_range:
            # labels_temp = np.ones((caps.size(0),))*class_curr
            # labels_temp = Variable(torch.LongTensor(labels_temp).cuda())
            recon_curr = model.just_reconstruct(caps_unit*mag_curr,labels)
            # print recon_curr.size()
            recons_all.append(recon_curr)

        # print caps.size()
        # raw_input()
        classes, reconstructions_gt, caps = [val.data.cpu().numpy() for val in [classes, reconstructions_gt,caps]]

        recons_all = [val.data.cpu().numpy() for val in recons_all]





        labels = labels.data.cpu().numpy()
        
        preds = np.argmax(classes,1)
        # print preds.shape, labels.shape
        # print np.sum(preds==labels)/float(labels.size)

        batch_size = data.shape[0]

        data = data.data.cpu().numpy()


        
        for im_num in range(batch_size):
            post_pend = []
            im_out = []

            gt_label = labels[im_num]
            pred_label = preds[im_num]
            # print 'gt',gt_label,'pred',pred_label
            im_in = data[im_num]
            # [0]
            # print im_in.shape
            # raw_input()
            im_in = (im_in*std_im)+mean_im
            # print im_in.shape
            im_out.append(im_in)
            post_pend.append(['org'])

            # recon_gt = reconstructions_gt[im_num]
            # # [0]
            # recon_gt = (recon_gt*std_im)+mean_im
            # # print recon_gt.shape
            # im_out.append(recon_gt)
            # post_pend.append(['recon_gt'])
            # routes_im = [np.sum(route[:,im_num,:,:],2) for route in routes]
            # # for val in im_out:
            # #     print val.shape
            
            for label_curr in range(len(recons_all)):
                recon_rel = np.array(recons_all[label_curr][im_num])
                recon_rel = (recon_rel*std_im)+mean_im
                # recon_rel = recon_rel+np.min(recon_rel)
                # recon_rel = recon_rel/np.max(recon_rel)
                im_out.append(recon_rel)
                post_pend.append([label_curr])


            pre_vals = [num_iter,im_num]
            # ,gt_label,pred_label]
            ims_row = save_all_im(out_dir_im,pre_vals,im_out,post_pend)
            ims_all.append(ims_row)
            # print ims_all
            # print len(ims_all)
            # print len(ims_all[0])
            # raw_input()
        # break
    
    # mats_to_save = []
    # mats_to_save = [labels,preds,routes[0],routes[1]]
    # mats_names = ['labels','preds','routes_0','routes_1']
    # for mat_curr, file_curr in zip(mats_to_save,mats_names):
    #     out_file_curr = os.path.join(out_dir_results,file_curr+'.npy')
    #     np.save(out_file_curr,mat_curr)

    np.save(os.path.join(out_dir_results,'ims_all.npy'),np.array(ims_all))


def save_class_as_other(out_dir_train,
                model_num,
                train_data,
                test_data,
                gpu_id = 0,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss(),
                margin_params = None,
                network_params = None,
                barebones = True,
                class_rel = 0,
                ):
    
    mag_range = np.arange(-0.5,0.6,0.1)
    out_dir_results = os.path.join(out_dir_train,'save_class_as_other_single_batch_'+str(model_num))
    util.makedirs(out_dir_results)
    out_dir_im = os.path.join(out_dir_results,'im_save')
    util.mkdir(out_dir_im)

    print out_dir_results
    
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
        
        # labels = batch['label'].cpu().numpy()
        # data = batch['image'].cpu().numpy()
        # rel_vals = labels==class_rel
        # labels = labels[rel_vals]
        # data = data[rel_vals]
        # batch['image'] = torch.Tensor(data)
        # batch['label'] = torch.LongTensor(labels)
        # print labels.shape
        # print data.shape

        # raw_input()

        # batch = test_dataloader.next() 
        if criterion=='marginmulti':
            labels = Variable(batch['label'].float().cuda())
        else:
            labels = Variable(torch.LongTensor(batch['label']).cuda())
        # labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())

        recons_all[(0,0)] = data.data.cpu().numpy()
        # labels = Variable(torch.LongTensor(batch['label']).cuda())
        
        # x = model.features(data)
        # _,routes = model.caps.forward_intrusive(x)
        # routes = [np.squeeze(routes_curr) for routes_curr in routes]
        # print len(routes), routes[0].shape


        # output, caps = 
        classes,reconstructions_gt,_,caps = model(data, labels, return_caps = True)
        
        caps_mag = (caps ** 2).sum(dim=-1) ** 0.5
        caps_unit = caps/caps_mag.view(caps_mag.size(0),caps_mag.size(1),1)

        recons_all = []
        for class_curr in range(caps.size(1)):
            labels_temp = np.ones((caps.size(0),))*class_curr
            labels_temp = Variable(torch.LongTensor(labels_temp).cuda())
            recon_curr = model.just_reconstruct(caps,labels_temp)
            # print recon_curr.size()
            recons_all.append(recon_curr)

        # print caps.size()
        # raw_input()
        classes, reconstructions_gt, caps = [val.data.cpu().numpy() for val in [classes, reconstructions_gt,caps]]

        recons_all = [val.data.cpu().numpy() for val in recons_all]





        labels = labels.data.cpu().numpy()
        
        preds = np.argmax(classes,1)
        # print preds.shape, labels.shape
        # print np.sum(preds==labels)/float(labels.size)

        batch_size = data.shape[0]

        data = data.data.cpu().numpy()


        ims_all = []
        for im_num in range(batch_size):
            post_pend = []
            im_out = []

            gt_label = labels[im_num]
            pred_label = preds[im_num]
            # print 'gt',gt_label,'pred',pred_label
            im_in = data[im_num]
            # [0]
            # print im_in.shape
            # raw_input()
            im_in = (im_in*std_im)+mean_im
            # print im_in.shape
            im_out.append(im_in)
            post_pend.append(['org'])

            # recon_gt = reconstructions_gt[im_num]
            # # [0]
            # recon_gt = (recon_gt*std_im)+mean_im
            # # print recon_gt.shape
            # im_out.append(recon_gt)
            # post_pend.append(['recon_gt'])
            # routes_im = [np.sum(route[:,im_num,:,:],2) for route in routes]
            # # for val in im_out:
            # #     print val.shape
            
            for label_curr in range(len(recons_all)):
                recon_rel = np.array(recons_all[label_curr][im_num])
                recon_rel = (recon_rel*std_im)+mean_im
                # recon_rel = recon_rel+np.min(recon_rel)
                # recon_rel = recon_rel/np.max(recon_rel)
                im_out.append(recon_rel)
                post_pend.append([label_curr])


            pre_vals = [im_num,gt_label,pred_label]
            ims_row = save_all_im(out_dir_im,pre_vals,im_out,post_pend)
            ims_all.append(ims_row)
            # print ims_all
            # print len(ims_all)
            # print len(ims_all[0])
            # raw_input()
        break
    
    # mats_to_save = []
    # mats_to_save = [labels,preds,routes[0],routes[1]]
    # mats_names = ['labels','preds','routes_0','routes_1']
    # for mat_curr, file_curr in zip(mats_to_save,mats_names):
    #     out_file_curr = os.path.join(out_dir_results,file_curr+'.npy')
    #     np.save(out_file_curr,mat_curr)

    np.save(os.path.join(out_dir_results,'ims_all.npy'),np.array(ims_all))
        


def save_class_vary_attr(out_dir_train,
                model_num,
                train_data,
                test_data,
                gpu_id = 0,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss(),
                margin_params = None,
                network_params = None,
                barebones = True,
                class_rel = 0,
                au = False
                ):
    
    mag_range = np.arange(-0.5,0.6,0.1)
    out_dir_results = os.path.join(out_dir_train,'save_class_vary_attr_single_batch_'+str(model_num)+'_'+str(class_rel))
    util.makedirs(out_dir_results)
    out_dir_im = os.path.join(out_dir_results,'im_save_'+str(class_rel))
    util.mkdir(out_dir_im)

    print out_dir_results
    
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

    ims_all = []
    for num_iter,batch in enumerate(test_dataloader):
        print 'NUM_ITER',num_iter
        
        labels = batch['label'].cpu().numpy()
        data = batch['image'].cpu().numpy()
        if au:
            print labels.shape
            rel_vals = labels[:,class_rel]>0
            print np.sum(rel_vals)
            labels = np.zeros((np.sum(rel_vals),labels.shape[1]))
            labels[:,class_rel]=1
            print labels.shape
            print rel_vals.shape
            
        else:
            rel_vals = labels==class_rel
            labels = labels[rel_vals]
        data = data[rel_vals]
        batch['image'] = torch.Tensor(data)
        batch['label'] = torch.LongTensor(labels)
        print labels.shape
        print data.shape
        
        if data.shape[0]==1:
            continue
        # raw_input()

        # batch = test_dataloader.next() 
        if criterion=='marginmulti':
            labels = Variable(batch['label'].float().cuda())
        else:
            labels = Variable(torch.LongTensor(batch['label']).cuda())
        # labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())

        # recons_all[(0,0)] = data.data.cpu().numpy()
        # labels = Variable(torch.LongTensor(batch['label']).cuda())
        
        # x = model.features(data)
        # _,routes = model.caps.forward_intrusive(x)
        # routes = [np.squeeze(routes_curr) for routes_curr in routes]
        # print len(routes), routes[0].shape


        # output, caps = 
        classes,reconstructions_gt,_,caps = model(data, labels, return_caps = True)
        
        caps_mag = (caps ** 2).sum(dim=-1) ** 0.5
        caps_unit = caps/caps_mag.view(caps_mag.size(0),caps_mag.size(1),1)

        # recons_all_all = []
        recons_all = []
        for attr_num in range(32):
            for mag_curr in mag_range:
                caps_curr = caps_unit.clone()
                caps_curr[:,:,attr_num]=mag_curr
                caps_mag_curr = (caps_curr ** 2).sum(dim=-1) ** 0.5
                caps_curr = caps_curr/caps_mag_curr.view(caps_mag_curr.size(0),caps_mag_curr.size(1),1)
                caps_curr = caps_curr* caps_mag.view(caps_mag.size(0),caps_mag.size(1),1)
                recon_curr = model.just_reconstruct(caps_curr,labels)
                recons_all.append(recon_curr)
            # recons_all_all.append(recons_all)

        # print caps.size()
        # raw_input()
        classes, reconstructions_gt, caps = [val.data.cpu().numpy() for val in [classes, reconstructions_gt,caps]]

        recons_all = [val.data.cpu().numpy() for val in recons_all]





        labels = labels.data.cpu().numpy()
        
        preds = np.argmax(classes,1)
        # print preds.shape, labels.shape
        # print np.sum(preds==labels)/float(labels.size)

        batch_size = data.shape[0]

        data = data.data.cpu().numpy()

        # ims_all = []
        # for label_curr in range(8):
        #     rel_idx = np.where(labels==label_curr)
        #     print rel_idx



        # ims_all_all = [[] for i in range(8)]
        # ims_all = []
        for im_num in range(batch_size):
            for attr_num in range(32):
                post_pend = []
                im_out = []

                gt_label = labels[im_num]
                pred_label = preds[im_num]

                # ims_all = ims_all_all[gt_label]
                # print 'gt',gt_label,'pred',pred_label
                im_in = data[im_num]
                # [0]
                # print im_in.shape
                # raw_input()
                im_in = (im_in*std_im)+mean_im
                # print im_in.shape
                im_out.append(im_in)
                post_pend.append(['org'])

                # recon_gt = reconstructions_gt[im_num]
                # # [0]
                # recon_gt = (recon_gt*std_im)+mean_im
                # # print recon_gt.shape
                # im_out.append(recon_gt)
                # post_pend.append(['recon_gt'])
                # routes_im = [np.sum(route[:,im_num,:,:],2) for route in routes]
                # # for val in im_out:
                # #     print val.shape
                
                # for label_curr in range(len(recons_all)):
            
                im_out.append(im_in)
                post_pend.append(['org'])

                for idx_mag_curr,mag_curr in enumerate(mag_range):
                    idx_curr = attr_num*len(mag_range)+idx_mag_curr
                    # print idx_curr,len(recons_all)
                    recon_rel = np.array(recons_all[idx_curr][im_num])
                    recon_rel = (recon_rel*std_im)+mean_im
                    # recon_rel = recon_rel+np.min(recon_rel)
                    # recon_rel = recon_rel/np.max(recon_rel)
                    im_out.append(recon_rel)
                    post_pend.append([attr_num,idx_mag_curr])


                # pre_vals = [im_num,gt_label,pred_label]
                pre_vals = [num_iter,im_num]
                ims_row = save_all_im(out_dir_im,pre_vals,im_out,post_pend)
                ims_all.append(ims_row)
            # print ims_all
            # print len(ims_all)
            # print len(ims_all[0])
            # raw_input()
        break
    
    # mats_to_save = []
    # mats_to_save = [labels,preds,routes[0],routes[1]]
    # mats_names = ['labels','preds','routes_0','routes_1']
    # for mat_curr, file_curr in zip(mats_to_save,mats_names):
    #     out_file_curr = os.path.join(out_dir_results,file_curr+'.npy')
    #     np.save(out_file_curr,mat_curr)

    np.save(os.path.join(out_dir_results,'ims_all.npy'),np.array(ims_all))


def save_class_vary_mag_class_rel(out_dir_train,
                model_num,
                train_data,
                test_data,
                gpu_id = 0,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss(),
                margin_params = None,
                network_params = None,
                barebones = True,
                class_rel = 0,
                au = False
                ):
    
    mag_range = np.arange(0.1,1.0,0.1)
    out_dir_results = os.path.join(out_dir_train,'save_class_vary_mag_single_batch_'+str(model_num)+'_'+str(class_rel))
    util.makedirs(out_dir_results)
    out_dir_im = os.path.join(out_dir_results,'im_save_'+str(class_rel))
    util.mkdir(out_dir_im)

    print out_dir_results
    
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

    ims_all = []
    for num_iter,batch in enumerate(test_dataloader):
        print 'NUM_ITER',num_iter
        
        labels = batch['label'].cpu().numpy()
        data = batch['image'].cpu().numpy()
        if au:
            print labels.shape
            rel_vals = labels[:,class_rel]>0
            print np.sum(rel_vals)
            labels = np.zeros((np.sum(rel_vals),labels.shape[1]))
            labels[:,class_rel]=1
            print labels.shape
            print rel_vals.shape
            
        else:
            rel_vals = labels==class_rel
            labels = labels[rel_vals]
        data = data[rel_vals]
        batch['image'] = torch.Tensor(data)
        batch['label'] = torch.LongTensor(labels)
        print labels.shape
        print data.shape
        
        if data.shape[0]==1:
            continue
        
        # labels = batch['label'].cpu().numpy()
        # data = batch['image'].cpu().numpy()
        # rel_vals = labels==class_rel
        # labels = labels[rel_vals]
        # data = data[rel_vals]
        # batch['image'] = torch.Tensor(data)
        # batch['label'] = torch.LongTensor(labels)
        # print labels.shape
        # print data.shape

        # raw_input()

        # batch = test_dataloader.next() 
        if criterion=='marginmulti':
            labels = Variable(batch['label'].float().cuda())
        else:
            labels = Variable(torch.LongTensor(batch['label']).cuda())
        # labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())

        # recons_all[(0,0)] = data.data.cpu().numpy()
        # labels = Variable(torch.LongTensor(batch['label']).cuda())
        
        # x = model.features(data)
        # _,routes = model.caps.forward_intrusive(x)
        # routes = [np.squeeze(routes_curr) for routes_curr in routes]
        # print len(routes), routes[0].shape


        # output, caps = 
        classes,reconstructions_gt,_,caps = model(data, labels, return_caps = True)
        
        caps_mag = (caps ** 2).sum(dim=-1) ** 0.5
        caps_unit = caps/caps_mag.view(caps_mag.size(0),caps_mag.size(1),1)

        recons_all = []
        for mag_curr in mag_range:
            # labels_temp = np.ones((caps.size(0),))*class_curr
            # labels_temp = Variable(torch.LongTensor(labels_temp).cuda())
            recon_curr = model.just_reconstruct(caps_unit*mag_curr,labels)
            # print recon_curr.size()
            recons_all.append(recon_curr)

        # print caps.size()
        # raw_input()
        classes, reconstructions_gt, caps = [val.data.cpu().numpy() for val in [classes, reconstructions_gt,caps]]

        recons_all = [val.data.cpu().numpy() for val in recons_all]





        labels = labels.data.cpu().numpy()
        
        preds = np.argmax(classes,1)
        # print preds.shape, labels.shape
        # print np.sum(preds==labels)/float(labels.size)

        batch_size = data.shape[0]

        data = data.data.cpu().numpy()


        
        for im_num in range(batch_size):
            post_pend = []
            im_out = []

            gt_label = labels[im_num]
            pred_label = preds[im_num]
            # print 'gt',gt_label,'pred',pred_label
            im_in = data[im_num]
            # [0]
            # print im_in.shape
            # raw_input()
            im_in = (im_in*std_im)+mean_im
            # print im_in.shape
            im_out.append(im_in)
            post_pend.append(['org'])

            # recon_gt = reconstructions_gt[im_num]
            # # [0]
            # recon_gt = (recon_gt*std_im)+mean_im
            # # print recon_gt.shape
            # im_out.append(recon_gt)
            # post_pend.append(['recon_gt'])
            # routes_im = [np.sum(route[:,im_num,:,:],2) for route in routes]
            # # for val in im_out:
            # #     print val.shape
            
            for label_curr in range(len(recons_all)):
                recon_rel = np.array(recons_all[label_curr][im_num])
                recon_rel = (recon_rel*std_im)+mean_im
                # recon_rel = recon_rel+np.min(recon_rel)
                # recon_rel = recon_rel/np.max(recon_rel)
                im_out.append(recon_rel)
                post_pend.append([label_curr])


            pre_vals = [num_iter,im_num]
            # ,gt_label,pred_label]
            ims_row = save_all_im(out_dir_im,pre_vals,im_out,post_pend)
            ims_all.append(ims_row)
            # print ims_all
            # print len(ims_all)
            # print len(ims_all[0])
            # raw_input()
        # break
    
    # mats_to_save = []
    # mats_to_save = [labels,preds,routes[0],routes[1]]
    # mats_names = ['labels','preds','routes_0','routes_1']
    # for mat_curr, file_curr in zip(mats_to_save,mats_names):
    #     out_file_curr = os.path.join(out_dir_results,file_curr+'.npy')
    #     np.save(out_file_curr,mat_curr)

    np.save(os.path.join(out_dir_results,'ims_all.npy'),np.array(ims_all))


