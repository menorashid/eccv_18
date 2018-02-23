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

class Exp_Lr_Scheduler:
    def __init__(self, optimizer,step_curr, init_lr, decay_rate, decay_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.step_curr = step_curr
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        

    def step(self):
        self.step_curr += 1
        for idx_param_group,param_group in enumerate(self.optimizer.param_groups): 
            # print idx_param_group,param_group['lr'],
            if self.init_lr[idx_param_group]!=0:
                new_lr = self.init_lr[idx_param_group] * self.decay_rate **(self.step_curr/self.decay_steps)
                new_lr = max(new_lr ,self.min_lr)
                param_group['lr'] = new_lr
            # print param_group['lr']

def train_model(out_dir_train,
                train_data,
                test_data,
                batch_size = None,
                batch_size_val =None,
                num_epochs = 100,
                save_after = 20,
                disp_after = 1,
                plot_after = 10,
                test_after = 1,
                lr = 0.0001,
                dec_after = 100, 
                model_name = 'alexnet',
                criterion = nn.CrossEntropyLoss(),
                gpu_id = 0,
                num_workers = 0,
                model_file = None,
                epoch_start = 0,
                margin_params = None,
                network_params = None,
                just_encoder = False):

    util.mkdir(out_dir_train)
    log_file = os.path.join(out_dir_train,'log.txt')
    plot_file = os.path.join(out_dir_train,'loss.jpg')
    log_arr = []
    plot_arr = [[],[]]
    plot_val_arr = [[],[]]

    network = models.get(model_name,network_params)
    # data_transforms = network.data_transforms
    if model_file is not None:
    #     model = network.model
    # else:
        if network_params is not None and just_encoder:
            network.model.features = torch.load(model_file).features
        else:
            network.model = torch.load(model_file)
    model = network.model

    # print model

    # train_data = dataset(train_file,data_transforms['train'])
    # test_data = dataset(test_file,data_transforms['val'])
    
    if batch_size is None:
        batch_size = len(train_data)

    if batch_size_val is None:
        batch_size_val = len(test_data)

    train_dataloader = torch.utils.data.DataLoader(train_data, 
                        batch_size=batch_size,
                        shuffle=True, 
                        num_workers=0)
    
    test_dataloader = torch.utils.data.DataLoader(test_data, 
                        batch_size=batch_size_val,
                        shuffle=False, 
                        num_workers=num_workers)
    
    torch.cuda.device(gpu_id)
    
    model = model.cuda()
    model.train(True)
    
    # optimizer = optim.SGD(network.get_lr_list(lr), lr=0, momentum=0.9)
    optimizer = torch.optim.Adam(network.get_lr_list(lr))
    print dec_after
    if dec_after is not None:
        print dec_after
        if dec_after[0] is 'step':
            print dec_after
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=dec_after[1], gamma=dec_after[2])
        elif dec_after[0] is 'exp':
            print dec_after
            exp_lr_scheduler = Exp_Lr_Scheduler(optimizer,epoch_start*len(train_dataloader),lr,dec_after[1],dec_after[2],dec_after[3])
        elif dec_after[0] is 'reduce':
            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=dec_after[1], factor=dec_after[2], patience=dec_after[3],min_lr=dec_after[4])
            
    if criterion=='spread':
        # margin = margin_params['start']
        criterion = Spread_Loss(**margin_params)

    recons = False
    # print network_params
    if model_name.startswith('dynamic_capsules') and network_params['reconstruct']:
        recons=True

    for num_epoch in range(epoch_start,num_epochs):

        # if isinstance(criterion,Spread_Loss):
        #     if num_epoch % margin_params['step'] ==0:
        #         i = num_epoch//margin_params['step']
        #         inc =  (1-margin_params['start'])/float(num_epochs//margin_params['step'])
        #         margin = i*inc+margin_params['start']

        for num_iter_train,batch in enumerate(train_dataloader):
            
            data = Variable(batch['image'].cuda())
            labels = Variable(torch.LongTensor(batch['label']).cuda())
            optimizer.zero_grad()

            if isinstance(criterion,Spread_Loss):
                loss = criterion(model(data), labels, num_epoch) 
            elif criterion =='margin' and recons:
                # print 'RECONS'
                loss = model.margin_loss(model(data,labels), labels) 
            elif criterion =='margin':
                loss = model.margin_loss(model(data), labels) 
            else:    
                loss = criterion(model(data), labels)    
            

            loss_iter = loss.data[0]
            loss.backward()
            optimizer.step()
            # if dec_after is not None and dec_after[0]=='exp':
            #     exp_lr_scheduler.step()
            
            num_iter = num_epoch*len(train_dataloader)+num_iter_train
            plot_arr[0].append(num_iter); plot_arr[1].append(loss_iter)
            if isinstance(criterion,Spread_Loss):
                str_display = 'margin: %.3f, lr: %.6f, iter: %d, loss: %.4f' %(criterion.margin,optimizer.param_groups[-1]['lr'],num_iter,loss_iter)    
            else:
                str_display = 'lr: %.6f, iter: %d, loss: %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter)
            log_arr.append(str_display)
            print str_display

            if num_iter % plot_after== 0 and num_iter>0:
                util.writeFile(log_file, log_arr)
                if len(plot_val_arr[0])==0:
                    visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
                else:
                    visualize.plotSimple([(plot_arr[0],plot_arr[1]),(plot_val_arr[0],plot_val_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train','Val'])


        if num_epoch % test_after == 0:
            model.eval()
            predictions = []
            labels_all = []
            loss_epoch = []
            for num_iter_test,batch in enumerate(test_dataloader):
                labels_all.append(batch['label'].numpy())
        
                data = Variable(batch['image'].cuda())
                labels = Variable(torch.LongTensor(batch['label']).cuda())
                output = model(data)
                
                if isinstance(criterion,Spread_Loss):
                    loss = criterion(output, labels, num_epoch) 
                    # model.spread_loss(output, labels, margin)    
                elif criterion =='margin':
                    loss = model.margin_loss(output, labels) 
                else:    
                    loss = criterion(output, labels)    
                loss_iter = loss.data[0]
                
                if isinstance(output, tuple):
                    out = output[0].data.cpu().numpy()
                else:
                    out = output.data.cpu().numpy()
                predictions.append(np.argmax(out,1))                
                
                loss_epoch.append(loss_iter)

            loss_iter = np.mean(loss_epoch)
            
            num_iter = num_epoch*len(train_dataloader)+len(train_dataloader)

            plot_val_arr[0].append(num_iter); plot_val_arr[1].append(loss_iter)

            str_display = 'lr: %.6f, val iter: %d, val loss: %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter)

            log_arr.append(str_display)
            print str_display
            labels_all = np.concatenate(labels_all)
            predictions = np.concatenate(predictions)
            accuracy = np.sum(predictions==labels_all)/float(labels_all.size)
            str_display = 'val accuracy: %.4f' %(accuracy)
            log_arr.append(str_display)
            print str_display
            

            model.train(True)

        if num_epoch % save_after == 0:
            out_file = os.path.join(out_dir_train,'model_'+str(num_epoch)+'.pt')
            print 'saving',out_file
            torch.save(model,out_file)

        if dec_after is not None and dec_after[0]=='reduce':
            # exp_lr_scheduler
            exp_lr_scheduler.step(loss_iter)
        elif dec_after is not None :
        # and dec_after[0]!='exp':
            exp_lr_scheduler.step()
    
    out_file = os.path.join(out_dir_train,'model_'+str(num_epoch)+'.pt')
    print 'saving',out_file
    torch.save(model,out_file)
    
    # print plot_arr[0]

    util.writeFile(log_file, log_arr)
    if len(plot_val_arr[0])==0:
        visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
    else:
        visualize.plotSimple([(plot_arr[0],plot_arr[1]),(plot_val_arr[0],plot_val_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train','Val'])   

  
def save_output_capsules(out_dir_train,
                model_num,
                train_data,
                test_data,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss(),
                network_params = None):

    out_dir_results = os.path.join(out_dir_train,'out_caps_model_test_'+str(model_num))
    print out_dir_results
    util.mkdir(out_dir_results)
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    log_file = os.path.join(out_dir_results,'log.txt')
    log_arr=[]

    # network = models.get(model_name)
    network = models.get(model_name,network_params)
    # data_transforms = network.data_transforms

    # test_data = dataset(test_file,data_transforms['val'])
    
    if batch_size_val is None:
        batch_size_val = len(test_data)
    

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_val,
                        shuffle=False, num_workers=1)

    torch.cuda.device(0)
    iter_begin = 0
    model = torch.load(model_file)
    model.cuda()
    model.eval()
    
    # criterion = nn.CrossEntropyLoss()
    
    predictions = []
    labels_all = []
    out_all = []
    caps_all = []
    recons_all = []

    for num_iter,batch in enumerate(test_dataloader):
                
        # batch = test_dataloader.next() 
        labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())
        labels = Variable(torch.LongTensor(batch['label']).cuda())
        

        # output, caps = 
        all_out = model(data, return_caps = True)
        caps = all_out[-1]    
        output = all_out[0]
        if len(all_out)>2:
            recons = all_out[1]
            recons_all.append(recons.data.cpu().numpy())

        out = output.data.cpu().numpy()
        # print out.shape
        caps = caps.data.cpu().numpy()
        # print caps.shape

        # raw_input()

        out_all.append(out)
        caps_all.append(caps)
        predictions.append(np.argmax(out,1))
        # if isinstance(criterion,Spread_Loss):
        #     loss = model.spread_loss(model(data), labels, margin)    
        # else:    
        loss = criterion(output, labels)    

        loss_iter = loss.data[0]

        str_display = 'iter: %d, val loss: %.4f' %(num_iter,loss_iter)
        log_arr.append(str_display)
        print str_display
        

        util.writeFile(log_file, log_arr)
    

    out_all = np.concatenate(out_all,0)
    predictions = np.concatenate(predictions)
    labels_all = np.concatenate(labels_all)
    caps_all = np.concatenate(caps_all,0)
    if len(recons_all)>0:
        recons_all = np.concatenate(recons_all,0)
        np.save(os.path.join(out_dir_results, 'recons_all.npy'),recons_all)    
    
    np.save(os.path.join(out_dir_results, 'out_all.npy'),out_all)
    np.save(os.path.join(out_dir_results, 'predictions.npy'),predictions)
    np.save(os.path.join(out_dir_results, 'labels_all.npy'),labels_all)
    np.save(os.path.join(out_dir_results, 'caps_all.npy'),caps_all)

    print labels_all.shape
    print predictions.shape
    print out_all.shape
    print caps_all.shape

    # y_true = np.zeros((labels_all.shape[0],2))
    # y_true[labels_all==0,0]=1
    # y_true[labels_all==1,1]=1

    # f1 = sklearn.metrics.f1_score(labels_all, predictions)
    # ap = sklearn.metrics.average_precision_score(y_true, out_all)
    # roc_auc = sklearn.metrics.roc_auc_score(y_true, out_all, average='macro')
    accuracy = np.sum(predictions==labels_all)/float(labels_all.size)

    # str_display = 'f1: %.4f' %(f1)
    # print str_display
    # log_arr.append(str_display)
    
    # str_display = 'ap: %.4f' %(ap)
    # print str_display
    # log_arr.append(str_display)
    
    # str_display = 'roc_auc: %.4f' %(roc_auc)
    # print str_display
    # log_arr.append(str_display)
    
    str_display = 'accuracy: %.4f' %(accuracy)
    print str_display
    log_arr.append(str_display)
    
    util.writeFile(log_file, log_arr)


def test_model(out_dir_train,
                model_num,
                train_data,
                test_data,
                gpu_id,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss(),
                margin_params  = None,
                network_params = None,
                post_pend = ''):

    out_dir_results = os.path.join(out_dir_train,'results_model_'+str(model_num)+post_pend)
    util.mkdir(out_dir_results)
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    log_file = os.path.join(out_dir_results,'log.txt')
    log_arr=[]

    # network = models.get(model_name)
    # network = models.get(model_name,network_params)
    # data_transforms = network.data_transforms

    # test_data = dataset(test_file,data_transforms['val'])
    
    if batch_size_val is None:
        batch_size_val = len(test_data)
    

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_val,
                        shuffle=False, num_workers=1)

    torch.cuda.device(gpu_id)
    iter_begin = 0
    model = torch.load(model_file)
    model.cuda()
    model.eval()
    
    # criterion = nn.CrossEntropyLoss()
    
    predictions = []
    labels_all = []
    out_all = []
    if criterion=='spread':
        criterion = Spread_Loss(**margin_params)
        

    for num_iter,batch in enumerate(test_dataloader):
                
        # batch = test_dataloader.next() 
        labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())
        labels = Variable(torch.LongTensor(batch['label']).cuda())
        

        output = model(data)
        out = output.data.cpu().numpy()
        out_all.append(out)
        
        predictions.append(np.argmax(out,1))
        if isinstance(criterion,Spread_Loss):
            loss = criterion(model(data), labels, model_num)    
        elif criterion=='margin':
            loss = model.margin_loss(model(data), labels) 
        else:    
            loss = criterion(output, labels)    

        loss_iter = loss.data[0]

        str_display = 'iter: %d, val loss: %.4f' %(num_iter,loss_iter)
        log_arr.append(str_display)
        print str_display
        

        util.writeFile(log_file, log_arr)
    
    out_all = np.concatenate(out_all,0)
    predictions = np.concatenate(predictions)
    labels_all = np.concatenate(labels_all)
    
    np.save(os.path.join(out_dir_results, 'out_all.npy'),out_all)
    np.save(os.path.join(out_dir_results, 'predictions.npy'),predictions)
    np.save(os.path.join(out_dir_results, 'labels_all.npy'),labels_all)
    
    accuracy = np.sum(predictions==labels_all)/float(labels_all.size)

    str_display = 'accuracy: %.4f' %(accuracy)
    print str_display
    log_arr.append(str_display)
    
    util.writeFile(log_file, log_arr)

def save_perturbed_images(out_dir_train,
                model_num,
                train_data,
                test_data,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss(),
                network_params = None):

    out_dir_results = os.path.join(out_dir_train,'vary_a_batch_squash_'+str(model_num))
    print out_dir_results
    util.mkdir(out_dir_results)
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    log_arr=[]

    network = models.get(model_name,network_params)
    
    if batch_size_val is None:
        batch_size_val = len(test_data)
    

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_val,
                        shuffle=False, num_workers=1)

    torch.cuda.device(0)
    iter_begin = 0
    model = torch.load(model_file)
    model.cuda()
    model.eval()
    
    # criterion = nn.CrossEntropyLoss()
    
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
        labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())
        labels = Variable(torch.LongTensor(batch['label']).cuda())
        

        # output, caps = 
        all_out = model(data, return_caps = True)
        caps = all_out[-1]    
        
        recons_all[(-1,-2)] = all_out[1].data.cpu().numpy()
        recons_all[(-1,-1)] = model.just_reconstruct(caps,labels).data.cpu().numpy()

        print caps.size()
        caps_data = caps.data.cpu().numpy()

        for dim_num in range(caps.size(2)):
            
            

            for inc_curr in np.arange(-0.25,0.30,0.05):
                caps = torch.autograd.Variable(torch.Tensor(caps_data)).cuda()
                caps[:,:,dim_num]=inc_curr
                squared_norm = (caps ** 2).sum(dim=2, keepdim=True)
                scale = squared_norm / (1 + squared_norm)
                caps = scale * caps / torch.sqrt(squared_norm)

                recons_curr = model.just_reconstruct(caps,labels)
                recons_all[(dim_num,inc_curr)]=recons_curr.data.cpu().numpy()
        # recons_curr = 
        break

    for key_curr in recons_all.keys():

        recons = recons_all[key_curr]
        recons = (recons*std_im)+mean_im
        out_dir = os.path.join(out_dir_results,'%d_%.2f'%(key_curr[0],key_curr[1]))
        util.mkdir(out_dir)
        for idx_im_curr,im_curr in enumerate(recons):
            scipy.misc.imsave(os.path.join(out_dir,str(idx_im_curr)+'.jpg'),im_curr[0])
        visualize.writeHTMLForFolder(out_dir,'.jpg')

