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
        # print 'STEPPING',len(self.optimizer.param_groups)
        for idx_param_group,param_group in enumerate(self.optimizer.param_groups): 
            # print 'outside',idx_param_group,self.init_lr[idx_param_group],param_group['lr']
            if self.init_lr[idx_param_group]!=0:
                new_lr = self.init_lr[idx_param_group] * self.decay_rate **(self.step_curr/float(self.decay_steps))
                new_lr = max(new_lr ,self.min_lr)
                # print idx_param_group,param_group['lr'], new_lr
                param_group['lr'] = new_lr
            # print param_group['lr']

def get_auc(pred,gt):

    # print pred.shape
    # print gt.shape
    # print pred
    # print gt
    # ap = []
    # gt[gt>0]=1
    # gt[gt<0]=0
    # print pred
    # print gt

    pred[pred>0.5]=1
    pred[pred<=0.5]=0
    # print pred

    ap = sklearn.metrics.f1_score(gt, pred,average='macro')

    # for idx in range(gt.shape[1]):
    #     ap = ap+[sklearn.metrics.average_precision_score(gt[:,idx], pred[:,idx])]
    return ap

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
                just_encoder = False,
                weight_decay = 0):

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
                        num_workers=num_workers)
    
    test_dataloader = torch.utils.data.DataLoader(test_data, 
                        batch_size=batch_size_val,
                        shuffle=False, 
                        num_workers=num_workers)
    
    torch.cuda.device(gpu_id)
    
    model = model.cuda()
    model.train(True)
    
    # optimizer = optim.SGD(network.get_lr_list(lr), lr=0, momentum=0.9)
    optimizer = torch.optim.Adam(network.get_lr_list(lr),weight_decay=weight_decay)
    print optimizer
    print dec_after
    if dec_after is not None:
        print dec_after
        if dec_after[0] is 'step':
            print dec_after
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=dec_after[1], gamma=dec_after[2])
        elif dec_after[0] is 'exp':
            print 'EXPING',dec_after
            exp_lr_scheduler = Exp_Lr_Scheduler(optimizer,epoch_start*len(train_dataloader),[lr_curr for lr_curr in lr if lr_curr!=0],dec_after[1],dec_after[2],dec_after[3])
        elif dec_after[0] is 'reduce':
            best_val = 0
            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=dec_after[1], factor=dec_after[2], patience=dec_after[3],min_lr=dec_after[4])
            
    if criterion=='spread':
        # margin = margin_params['start']
        criterion = Spread_Loss(**margin_params)

    recons = False
    # print network_params
    if model_name.endswith('recon') or (model_name.startswith('dynamic_capsules') and network_params['reconstruct']):
        recons=True

    print 'RECON',recons
    for num_epoch in range(epoch_start,num_epochs):

        # if isinstance(criterion,Spread_Loss):
        #     if num_epoch % margin_params['step'] ==0:
        #         i = num_epoch//margin_params['step']
        #         inc =  (1-margin_params['start'])/float(num_epochs//margin_params['step'])
        #         margin = i*inc+margin_params['start']

        for num_iter_train,batch in enumerate(train_dataloader):

            data = Variable(batch['image'].cuda())
            # print torch.min(data),torch.max(data)
            # raw_input()
            
            if isinstance(criterion,nn.MultiLabelSoftMarginLoss):
                labels = Variable(batch['label'].float().cuda())
            else:
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
                # out = model(data)
                # print torch.min(out),torch.max(out)
                loss = criterion(model(data), labels)    
            

            loss_iter = loss.data[0]
            loss.backward()
            optimizer.step()
            if dec_after is not None and dec_after[0]=='exp':
                exp_lr_scheduler.step()
            
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
                if isinstance(criterion,nn.MultiLabelSoftMarginLoss):
                    labels = Variable(batch['label'].float().cuda())
                else:
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
                if len(labels.size())>1:
                    predictions.append(out)
                else:
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

            print labels_all.shape,predictions.shape
            if len(labels_all.shape)>1:
                accuracy = get_auc(predictions,labels_all)
            else:
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
            if accuracy>=best_val:
                best_val = accuracy
                out_file_best = os.path.join(out_dir_train,'model_bestVal.pt')
                print 'saving',out_file_best
                torch.save(model,out_file_best)            
            exp_lr_scheduler.step(loss_iter)

        elif dec_after is not None and dec_after[0]!='exp':
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
                post_pend = '',
                model_nums = None):

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

    recons=False
    if model_name.endswith('recon') or (model_name.startswith('dynamic_capsules') and network_params['reconstruct']):
        recons=True

    print 'RECON',recons

    for num_iter,batch in enumerate(test_dataloader):
                
        # batch = test_dataloader.next() 
        labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())
        if isinstance(criterion,nn.MultiLabelSoftMarginLoss):
            labels = Variable(batch['label'].float().cuda())
        else:
            labels = Variable(torch.LongTensor(batch['label']).cuda())

        

        output = model(data)
        if recons:
            output = output[0]
        out = output.data.cpu().numpy()
        out_all.append(out)
        
        predictions.append(np.argmax(out,1))
        if isinstance(criterion,Spread_Loss):
            if isinstance(model_num,int):
                loss = criterion(model(data), labels, model_num)    
                loss_iter = loss.data[0]
            else:
                loss_iter = 0

        elif criterion=='margin':
            loss,a,b = model.margin_loss(model(data), labels) 

            loss_iter = loss.data[0]
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
    
    if len(labels_all.shape)>1:
        accuracy = get_auc(out_all,labels_all)
    else:
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

def test_model_list_models(out_dir_train,
                model_nums,
                train_data,
                test_data,
                gpu_id,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss(),
                margin_params  = None,
                network_params = None,
                post_pend = '',
                model_num=None):

    log_file_meta = os.path.join(out_dir_train,'log_test'+post_pend+'.txt')
    log_arr_meta=[]

    # network = models.get(model_name)
    # network = models.get(model_name,network_params)
    # data_transforms = network.data_transforms

    # test_data = dataset(test_file,data_transforms['val'])
    
    if batch_size_val is None:
        batch_size_val = len(test_data)
    

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_val,
                        shuffle=False, num_workers=1)

    torch.cuda.device(gpu_id)
    
    
    for model_num in model_nums:
        out_dir_results = os.path.join(out_dir_train,'results_model_'+str(model_num)+post_pend)
        util.mkdir(out_dir_results)
        log_file = os.path.join(out_dir_results,'log.txt')
        log_arr=[]

    
        model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
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
            if isinstance(criterion,nn.MultiLabelSoftMarginLoss):
                labels = Variable(batch['label'].float().cuda())
            else:
                labels = Variable(torch.LongTensor(batch['label']).cuda())

            

            output = model(data)
            out = output.data.cpu().numpy()
            out_all.append(out)
            
            predictions.append(np.argmax(out,1))
            if isinstance(criterion,Spread_Loss):
                if isinstance(model_num,int):
                    loss = criterion(model(data), labels, model_num)    
                    loss_iter = loss.data[0]
                else:
                    loss_iter = 0

            elif criterion=='margin':
                loss = model.margin_loss(model(data), labels) 
                loss_iter = loss.data[0]
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
        
        if len(labels_all.shape)>1:
            accuracy = get_auc(out_all,labels_all)
        else:
            accuracy = np.sum(predictions==labels_all)/float(labels_all.size)

        str_display = 'model: %s, val accuracy: %.4f' %(str(model_num),accuracy)
        print str_display
        log_arr.append(str_display)

        log_arr_meta = log_arr_meta+log_arr

        util.writeFile(log_file, log_arr)

    util.writeFile(log_file_meta, log_arr_meta)
    
    

def train_model_recon(out_dir_train,
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
                just_encoder = False,
                weight_decay = 0):

    util.mkdir(out_dir_train)
    log_file = os.path.join(out_dir_train,'log.txt')

    log_file_writer = open(log_file,'wb')
    
    plot_file = os.path.join(out_dir_train,'loss.jpg')
    log_arr = []
    plot_arr = [[[],[]],[[],[]],[[],[]]]
    plot_val_arr =  [[[],[]],[[],[]],[[],[]]]
    plot_val_acc_arr = [[],[]]
    plot_strs_posts = ['Total','Margin','Recon']
    plot_acc_file = os.path.join(out_dir_train,'val_accu.jpg')
    
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
                        num_workers=num_workers)
    
    test_dataloader = torch.utils.data.DataLoader(test_data, 
                        batch_size=batch_size_val,
                        shuffle=False, 
                        num_workers=num_workers)
    print 'GPU ID ',gpu_id
    torch.cuda.set_device(gpu_id)
    # cuda.device(gpu_id)
    
    model = model.cuda()
    model.train(True)
    print model
    
    # optimizer = optim.SGD(network.get_lr_list(lr), lr=0, momentum=0.9)
    optimizer = torch.optim.Adam(network.get_lr_list(lr),weight_decay=weight_decay)
    print optimizer
    print dec_after
    if dec_after is not None:
        print dec_after
        if dec_after[0] is 'step':
            print dec_after
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=dec_after[1], gamma=dec_after[2])
        elif dec_after[0] is 'exp':
            print dec_after
            exp_lr_scheduler = Exp_Lr_Scheduler(optimizer,epoch_start*len(train_dataloader),[lr_curr for lr_curr in lr if lr_curr!=0],dec_after[1],dec_after[2],dec_after[3])
        elif dec_after[0] is 'reduce':
            best_val = 0
            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=dec_after[1], factor=dec_after[2], patience=dec_after[3],min_lr=dec_after[4])
            
    if criterion=='spread':
        # margin = margin_params['start']
        criterion = Spread_Loss(**margin_params)

    
    recons=True

    print 'RECON',recons
    for num_epoch in range(epoch_start,num_epochs):

        # if isinstance(criterion,Spread_Loss):
        #     if num_epoch % margin_params['step'] ==0:
        #         i = num_epoch//margin_params['step']
        #         inc =  (1-margin_params['start'])/float(num_epochs//margin_params['step'])
        #         margin = i*inc+margin_params['start']

        for num_iter_train,batch in enumerate(train_dataloader):

            data = Variable(batch['image'].cuda())
            # print torch.min(data),torch.max(data)
            # print data.size()

            # raw_input()
            
            if criterion=='marginmulti':
                labels = Variable(batch['label'].float().cuda())
            else:
                labels = Variable(torch.LongTensor(batch['label']).cuda())
            optimizer.zero_grad()

            
            
            loss, margin_loss, recon_loss = model.margin_loss(model(data,labels), labels) 

            

            loss_iter = loss.data[0]
            loss.backward()
            optimizer.step()
            if dec_after is not None and dec_after[0]=='exp':
                exp_lr_scheduler.step()
            
            num_iter = num_epoch*len(train_dataloader)+num_iter_train
            for idx_loss,loss_curr in enumerate([loss, margin_loss, recon_loss]):
                plot_arr[idx_loss][0].append(num_iter)
                plot_arr[idx_loss][1].append(loss_curr.data)
            
            
            str_display = 'lr: %.6f, iter: %d, loss: %.4f, margin: %.4f, recon: %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter,margin_loss.data[0],recon_loss.data[0])



            log_arr.append(str_display)
            print str_display

            if num_iter % plot_after== 0 and num_iter>0:
                # util.writeFile(log_file, log_arr)
                for string in log_arr:
                    log_file_writer.write(string+'\n')
                log_arr = []

                if len(plot_val_arr[0])==0:
                    visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
                else:
                    
                    lengend_strs = [pre_str+plot_str_posts for pre_str in ['Train ','Test '] for plot_str_posts in plot_strs_posts]

                    plot_vals = [(arr[0],arr[1]) for arr in plot_arr+plot_val_arr]
                    # print plot_vals
                    # print lengend_strs
                    visualize.plotSimple(plot_vals,out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=lengend_strs)

                    visualize.plotSimple([(plot_val_acc_arr[0],plot_val_acc_arr[1])],out_file = plot_acc_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Accuracy',legend_entries=['Val'])


        if num_epoch % test_after == 0:
            model.eval()
            predictions = []
            labels_all = []
            loss_epoch = []
            margin_loss_epoch = []
            recon_loss_epoch = []
            for num_iter_test,batch in enumerate(test_dataloader):
                labels_all.append(batch['label'].numpy())
        
                data = Variable(batch['image'].cuda())
                if criterion=='marginmulti':
                    labels = Variable(batch['label'].float().cuda())
                else:
                    labels = Variable(torch.LongTensor(batch['label']).cuda())
                output = model(data)
                
                
                
                losses = model.margin_loss(output, labels) 
                losses = [loss_curr.data[0] for loss_curr in losses]
                loss_iter, margin_loss_iter, recon_loss_iter = losses

                
                
                if isinstance(output, tuple):
                    out = output[0].data.cpu().numpy()
                else:
                    out = output.data.cpu().numpy()
                if len(labels.size())>1:
                    predictions.append(out)
                else:
                    predictions.append(np.argmax(out,1))                
                
                loss_epoch.append(loss_iter)
                margin_loss_epoch.append(margin_loss_iter)
                recon_loss_epoch.append(recon_loss_iter)
                
                num_iter = num_epoch*len(train_dataloader)+len(train_dataloader)+num_iter_test
                str_display = 'lr: %.6f, val iter: %d, val loss: %.4f, margin: %.4f, recon: %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter,margin_loss_iter,recon_loss_iter)
                log_arr.append(str_display)
                print str_display

            loss_iter = np.mean(loss_epoch)
            margin_loss_iter = np.mean(margin_loss_epoch)
            recon_loss_iter = np.mean(recon_loss_epoch)
            num_iter = num_epoch*len(train_dataloader)+len(train_dataloader)

            for idx_loss,loss_curr in enumerate([loss_iter, margin_loss_iter, recon_loss_iter]):
                plot_val_arr[idx_loss][0].append(num_iter)
                plot_val_arr[idx_loss][1].append(loss_curr)
            

            str_display = 'lr: %.6f, val iter: %d, val loss: %.4f, margin: %.4f, recon %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter,margin_loss_iter,recon_loss_iter)

            log_arr.append(str_display)
            print str_display
            labels_all = np.concatenate(labels_all)
            predictions = np.concatenate(predictions)


            print labels_all.shape,predictions.shape
            print labels_all[:3,:]
            # pred_print = np.zeros(predictions.shape)
            # pred_print[:,np.argmax(predictions,1)]=1.
            print predictions[:3,:]

            if len(labels_all.shape)>1:
                accuracy = get_auc(predictions,labels_all)
            else:
                accuracy = np.sum(predictions==labels_all)/float(labels_all.size)

            plot_val_acc_arr[0].append(num_iter); plot_val_acc_arr[1].append(accuracy)
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
            if accuracy>=best_val:
                best_val = accuracy
                out_file_best = os.path.join(out_dir_train,'model_bestVal.pt')
                print 'saving',out_file_best
                torch.save(model,out_file_best)            
            exp_lr_scheduler.step(loss_iter)

        elif dec_after is not None and dec_after[0]!='exp':
            exp_lr_scheduler.step()
    
    out_file = os.path.join(out_dir_train,'model_'+str(num_epoch)+'.pt')
    print 'saving',out_file
    torch.save(model,out_file)
    
    # print plot_arr[0]

    # util.writeFile(log_file, log_arr)
    for string in log_arr:
        log_file_writer.write(string+'\n')
    log_arr = []

    if len(plot_val_arr[0])==0:
        visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
    else:
        lengend_strs = [pre_str+plot_str_posts for pre_str in ['Train ','Test '] for plot_str_posts in plot_strs_posts]

        plot_vals = [(arr[0],arr[1]) for arr in plot_arr+plot_val_arr]
        # print plot_vals
        # print lengend_strs
        visualize.plotSimple(plot_vals,out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=lengend_strs)
        visualize.plotSimple([(plot_val_acc_arr[0],plot_val_acc_arr[1])],out_file = plot_acc_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Accuracy',legend_entries=['Val'])

        # visualize.plotSimple([(plot_arr[0],plot_arr[1]),(plot_val_arr[0],plot_val_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train','Val'])   
    log_file_writer.close()

def train_model_recon_au(out_dir_train,
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
                just_encoder = False,
                weight_decay = 0):

    util.mkdir(out_dir_train)
    log_file = os.path.join(out_dir_train,'log.txt')
    plot_file = os.path.join(out_dir_train,'loss.jpg')
    log_arr = []
    plot_arr = [[[],[]],[[],[]],[[],[]],[[],[]]]
    plot_val_arr =  [[[],[]],[[],[]],[[],[]],[[],[]]]
    plot_val_acc_arr = [[],[]]
    plot_strs_posts = ['Total','Margin','Margin AU', 'Recon']
    plot_acc_file = os.path.join(out_dir_train,'val_accu.jpg')
    
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
    optimizer = torch.optim.Adam(network.get_lr_list(lr),weight_decay=weight_decay)
    print optimizer
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
            best_val = 0
            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=dec_after[1], factor=dec_after[2], patience=dec_after[3],min_lr=dec_after[4])
            
    if criterion=='spread':
        # margin = margin_params['start']
        criterion = Spread_Loss(**margin_params)

    
    recons=True

    print 'RECON',recons
    for num_epoch in range(epoch_start,num_epochs):

        # if isinstance(criterion,Spread_Loss):
        #     if num_epoch % margin_params['step'] ==0:
        #         i = num_epoch//margin_params['step']
        #         inc =  (1-margin_params['start'])/float(num_epochs//margin_params['step'])
        #         margin = i*inc+margin_params['start']

        for num_iter_train,batch in enumerate(train_dataloader):

            data = Variable(batch['image'].cuda())
            # print torch.min(data),torch.max(data)
            # raw_input()
            
            if isinstance(criterion,nn.MultiLabelSoftMarginLoss):
                labels = Variable(batch['label'].float().cuda())
            else:
                labels = Variable(torch.LongTensor(batch['label']).cuda())

            # print batch.keys()
            # for k in batch.keys():
            #     print k,batch[k].size()

            labels_au = Variable(batch['label_au'].float().cuda())
            bin_au = Variable(batch['bin_au'].float().cuda())

            optimizer.zero_grad()

            loss, margin_loss, margin_au_loss, recon_loss = model.margin_loss_multi(model(data,labels), labels, labels_au, bin_au) 

            

            loss_iter = loss.data[0]
            loss.backward()
            optimizer.step()
            if dec_after is not None and dec_after[0]=='exp':
                exp_lr_scheduler.step()
            
            num_iter = num_epoch*len(train_dataloader)+num_iter_train
            for idx_loss,loss_curr in enumerate([loss, margin_loss, margin_au_loss, recon_loss]):

                if idx_loss==2 and float(loss_curr.data)==0:
                    continue
                plot_arr[idx_loss][0].append(num_iter)
                plot_arr[idx_loss][1].append(loss_curr.data)
            
            
            str_display = 'lr: %.6f, iter: %d, loss: %.4f, margin: %.4f, margin_au: %.4f, recon: %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter,margin_loss.data[0],margin_au_loss.data[0],recon_loss.data[0])
            


            log_arr.append(str_display)
            print str_display

            if num_iter % plot_after== 0 and num_iter>0:
                util.writeFile(log_file, log_arr)
                if len(plot_val_arr[0])==0:
                    visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
                else:
                    
                    lengend_strs = [pre_str+plot_str_posts for pre_str in ['Train ','Test '] for plot_str_posts in plot_strs_posts]

                    plot_vals = [(arr[0],arr[1]) for arr in plot_arr+plot_val_arr]
                    # print plot_vals
                    # print lengend_strs
                    visualize.plotSimple(plot_vals,out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=lengend_strs)

                    visualize.plotSimple([(plot_val_acc_arr[0],plot_val_acc_arr[1])],out_file = plot_acc_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Accuracy',legend_entries=['Val'])


        if num_epoch % test_after == 0:
            model.eval()
            predictions = []
            labels_all = []
            loss_epoch = []
            margin_loss_epoch = []
            margin_au_loss_epoch = []
            recon_loss_epoch = []
            for num_iter_test,batch in enumerate(test_dataloader):
                labels_all.append(batch['label'].numpy())
        
                data = Variable(batch['image'].cuda())
                if isinstance(criterion,nn.MultiLabelSoftMarginLoss):
                    labels = Variable(batch['label'].float().cuda())
                else:
                    labels = Variable(torch.LongTensor(batch['label']).cuda())

                labels_au = Variable(batch['label_au'].float().cuda())
                bin_au = Variable(batch['bin_au'].float().cuda())

                output = model(data)
                
                loss, margin_loss, margin_au_loss, recon_loss = model.margin_loss_multi(output, labels, labels_au, bin_au) 
                
                loss_iter = loss.data[0]
                
                if isinstance(output, tuple):
                    out = output[0].data.cpu().numpy()
                else:
                    out = output.data.cpu().numpy()
                if len(labels.size())>1:
                    predictions.append(out)
                else:
                    predictions.append(np.argmax(out,1))                
                
                loss_epoch.append(loss_iter)
                margin_loss_epoch.append(margin_loss.data[0])
                margin_au_loss_epoch.append(margin_au_loss.data[0])
                recon_loss_epoch.append(recon_loss.data[0])

            loss_iter = np.mean(loss_epoch)
            margin_loss_iter = np.mean(margin_loss_epoch)
            margin_au_loss_iter = np.mean(margin_au_loss_epoch)
            recon_loss_iter = np.mean(recon_loss_epoch)
            
            num_iter = num_epoch*len(train_dataloader)+len(train_dataloader)

            for idx_loss,loss_curr in enumerate([loss_iter, margin_loss_iter, recon_loss_iter]):
                plot_val_arr[idx_loss][0].append(num_iter)
                plot_val_arr[idx_loss][1].append(loss_curr)
            

            str_display = 'lr: %.6f, val iter: %d, val loss: %.4f, margin: %.4f, margin_au: %.4f, recon %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter,margin_loss_iter,margin_au_loss_iter, recon_loss_iter)

            log_arr.append(str_display)
            print str_display
            labels_all = np.concatenate(labels_all)
            predictions = np.concatenate(predictions)

            # print labels_all.shape,predictions.shape
            if len(labels_all.shape)>1:
                accuracy = get_auc(predictions,labels_all)
            else:
                accuracy = np.sum(predictions==labels_all)/float(labels_all.size)

            plot_val_acc_arr[0].append(num_iter); plot_val_acc_arr[1].append(accuracy)
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
            if accuracy>=best_val:
                best_val = accuracy
                out_file_best = os.path.join(out_dir_train,'model_bestVal.pt')
                print 'saving',out_file_best
                torch.save(model,out_file_best)            
            exp_lr_scheduler.step(loss_iter)

        elif dec_after is not None and dec_after[0]!='exp':
            exp_lr_scheduler.step()
    
    out_file = os.path.join(out_dir_train,'model_'+str(num_epoch)+'.pt')
    print 'saving',out_file
    torch.save(model,out_file)
    
    # print plot_arr[0]

    util.writeFile(log_file, log_arr)
    if len(plot_val_arr[0])==0:
        visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
    else:
        lengend_strs = [pre_str+plot_str_posts for pre_str in ['Train ','Test '] for plot_str_posts in plot_strs_posts]

        plot_vals = [(arr[0],arr[1]) for arr in plot_arr+plot_val_arr]
        # print plot_vals
        # print lengend_strs
        visualize.plotSimple(plot_vals,out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=lengend_strs)
        visualize.plotSimple([(plot_val_acc_arr[0],plot_val_acc_arr[1])],out_file = plot_acc_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Accuracy',legend_entries=['Val'])

        # visualize.plotSimple([(plot_arr[0],plot_arr[1]),(plot_val_arr[0],plot_val_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train','Val'])   



def test_model_recon(out_dir_train,
                model_num,
                train_data,
                test_data,
                gpu_id,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss(),
                margin_params  = None,
                network_params = None,
                post_pend = '',
                model_nums = None,barebones = True):

    out_dir_results = os.path.join(out_dir_train,'results_model_'+str(model_num)+post_pend)
    util.mkdir(out_dir_results)
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    log_file = os.path.join(out_dir_results,'log.txt')
    log_arr=[]


    if batch_size_val is None:
        batch_size_val = len(test_data)
    

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_val,
                        shuffle=False, num_workers=0)
    
    print len(test_data)

    torch.cuda.device(gpu_id)
    iter_begin = 0
    model = torch.load(model_file)
    model.cuda()
    model.eval()

    predictions = []
    labels_all = []
    out_all = []
    recon_all = []

    recons=True

    print 'RECON',recons

    print 'barebones',barebones
    
    num_correct =0
    num_total = 0

    for num_iter,batch in enumerate(test_dataloader):
        predictions = []
        labels_all = []
        loss_epoch = []
        margin_loss_epoch = []
        recon_loss_epoch = []
        recon_all = []
        recon_all_gt = []
        caps_all = []
        im_org = []

        labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())
        # print num_iter, data.size()
        # continue

        if criterion=='marginmulti':
            labels = Variable(batch['label'].float().cuda())
        else:
            labels = Variable(torch.LongTensor(batch['label']).cuda())
        
        output = model(data, return_caps = False)
        # print output
        # print labels
        # raw_input()
        # losses = model.margin_loss(output[0], labels) 
        # losses = [loss_curr.data[0] for loss_curr in losses]
        # loss_iter, margin_loss_iter, recon_loss_iter = losses
        losses = model.margin_loss(output, labels) 
        losses = [loss_curr.data[0] for loss_curr in losses]
        loss_iter, margin_loss_iter, recon_loss_iter = losses
        
        if isinstance(output, tuple):
            

            output = list(output)
            if barebones:
                out = output[0].data.cpu().numpy()
            else:
                output = [val.data.cpu().numpy() for val in output]
                out, reconstructions, data_c, caps = output
                recon_all.append(reconstructions)
                im_org.append(data_c)
                caps_all.append(caps)

                output_gt = model(data,labels, return_caps = True)
                reconstructions_gt = output_gt[1]
                reconstructions_gt = reconstructions_gt.data.cpu().numpy()
                recon_all_gt.append(reconstructions_gt)
            
        else:
            out = output.data.cpu().numpy()
        


        if len(labels.size())>1:
            predictions.append(out)
        else:
            predictions.append(np.argmax(out,1))                
        
        loss_epoch.append(loss_iter)
        margin_loss_epoch.append(margin_loss_iter)
        recon_loss_epoch.append(recon_loss_iter)
        
        # num_iter = num_epoch*len(train_dataloader)+len(train_dataloader)
        str_display = 'val iter: %d, val loss: %.4f, margin: %.4f, recon: %.4f' %(num_iter,loss_iter,margin_loss_iter,recon_loss_iter)
        log_arr.append(str_display)
        print str_display
        
        # labels_all = np.concatenate(labels_all)
        labels_all = labels_all[0]
        # predictions = np.concatenate(predictions)
        predictions = predictions[0]

        num_correct = num_correct+ np.sum(predictions==labels_all)
        num_total = num_total+labels_all.size

        print os.path.join(out_dir_results,'labels_all_'+str(num_iter)+'.npy'),labels_all.shape
        np.save(os.path.join(out_dir_results,'labels_all_'+str(num_iter)+'.npy'),labels_all)
        np.save(os.path.join(out_dir_results,'predictions_'+str(num_iter)+'.npy'),predictions)
        
        if len(recon_all)>0:
            # recon_all = np.concatenate(recon_all,axis=0)
            recon_all = recon_all[0]
            # recon_all_gt = np.concatenate(recon_all_gt,axis=0)
            recon_all_gt = recon_all_gt[0]
            # caps_all = np.concatenate(caps_all,axis=0)
            caps_all = caps_all[0]
            # im_org = np.concatenate(im_org,axis=0)
            im_org = im_org[0]
            np.save(os.path.join(out_dir_results,'recon_all_'+str(num_iter)+'.npy'),recon_all)
            np.save(os.path.join(out_dir_results,'recon_all_gt_'+str(num_iter)+'.npy'),recon_all_gt)
            np.save(os.path.join(out_dir_results,'caps_all_'+str(num_iter)+'.npy'),caps_all)
            np.save(os.path.join(out_dir_results,'im_org_'+str(num_iter)+'.npy'),im_org)




    loss_iter = np.mean(loss_epoch)
    margin_loss_iter = np.mean(margin_loss_epoch)
    recon_loss_iter = np.mean(recon_loss_epoch)
    
    str_display = 'val iter: %d, val loss mean: %.4f, margin mean: %.4f, recon mean: %.4f' %(num_iter,loss_iter,margin_loss_iter,recon_loss_iter)
    log_arr.append(str_display)
    print str_display
        

    

    

    print out_dir_results




    # print labels_all.shape,predictions.shape
    if len(labels_all.shape)>1:
        accuracy = get_auc(predictions,labels_all)
    else:
        accuracy = num_correct/float(num_total)
        print num_correct,num_total
        # np.sum(predictions==labels_all)/float(labels_all.size)

    str_display = 'val accuracy: %.4f' %(accuracy)
    log_arr.append(str_display)
    print str_display
    

    util.writeFile(log_file, log_arr)
    



