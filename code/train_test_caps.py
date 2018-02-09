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
        new_lr = self.init_lr * self.decay_rate **(self.step_curr/self.decay_steps)
        new_lr = max(new_lr ,self.min_lr)

        for param_group in self.optimizer.param_groups:    
            param_group['lr'] = new_lr

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
                network_params = None):

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
        network.model = torch.load(model_file)
    model = network.model

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
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=dec_after[1], gamma=0.1)
        elif dec_after[0] is 'exp':
            print dec_after
            exp_lr_scheduler = Exp_Lr_Scheduler(optimizer,epoch_start*len(train_dataloader),lr,dec_after[1],dec_after[2],dec_after[3])
            

    if criterion=='spread':
        margin = margin_params['start']

    for num_epoch in range(epoch_start,num_epochs):

        if criterion=='spread':
            if num_epoch % margin_params['step'] ==0:
                i = num_epoch//margin_params['step']
                inc =  (1-margin_params['start'])/float(num_epochs//margin_params['step'])
                margin = i*inc+margin_params['start']

        for num_iter_train,batch in enumerate(train_dataloader):
            
            data = Variable(batch['image'].cuda())
            labels = Variable(torch.LongTensor(batch['label']).cuda())
            optimizer.zero_grad()

            if criterion=='spread':
                loss = model.spread_loss(model(data), labels, margin) 
            elif criterion =='margin':
                loss = model.margin_loss(model(data), labels) 
            else:    
                loss = criterion(model(data), labels)    
            

            loss_iter = loss.data[0]
            loss.backward()
            optimizer.step()
            if dec_after is not None and dec_after[0]=='exp':
                exp_lr_scheduler.step()
            
            num_iter = num_epoch*len(train_dataloader)+num_iter_train
            plot_arr[0].append(num_iter); plot_arr[1].append(loss_iter)
            if criterion=='spread':
                str_display = 'margin: %.3f, lr: %.6f, iter: %d, loss: %.4f' %(margin,optimizer.param_groups[-1]['lr'],num_iter,loss_iter)    
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


        if num_epoch % test_after == 0 :
            model.eval()
            predictions = []
            labels_all = []
    
            for num_iter_test,batch in enumerate(test_dataloader):
                labels_all.append(batch['label'].numpy())
        
                data = Variable(batch['image'].cuda())
                labels = Variable(torch.LongTensor(batch['label']).cuda())
                output = model(data)
                
                out = output.data.cpu().numpy()
                predictions.append(np.argmax(out,1))                
                if criterion=='spread':
                    loss = model.spread_loss(model(data), labels, margin)    
                elif criterion =='margin':
                    loss = model.margin_loss(model(data), labels) 
                else:    
                    loss = criterion(output, labels)    
                loss_iter = loss.data[0]

                num_iter = num_epoch*len(train_dataloader)+num_iter_test
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

        if dec_after is not None and dec_after[0]!='exp':
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

            

def test_model(out_dir_train,
                model_num,
                train_data,
                test_data,
                model_name = 'alexnet',
                batch_size_val =None,
                criterion = nn.CrossEntropyLoss()):

    out_dir_results = os.path.join(out_dir_train,'results_model_'+str(model_num))
    util.mkdir(out_dir_results)
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    log_file = os.path.join(out_dir_results,'log.txt')
    log_arr=[]

    network = models.get(model_name)
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

    for num_iter,batch in enumerate(test_dataloader):
                
        # batch = test_dataloader.next() 
        labels_all.append(batch['label'].numpy())

        data = Variable(batch['image'].cuda())
        labels = Variable(torch.LongTensor(batch['label']).cuda())
        

        output = model(data)
        out = output.data.cpu().numpy()
        out_all.append(out)
        
        predictions.append(np.argmax(out,1))
        if criterion=='spread':
            loss = model.spread_loss(model(data), labels, margin)    
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

def old_exp():
    out_dir_meta = '../experiments/mat_capsules/'
    num_epochs = 50
    dec_after = 50
    lr = 0.02
    split_num = 0
    margin_params = {'step':1,'start':0.2}

    strs_append = '_'.join([str(val) for val in [num_epochs,dec_after,lr,margin_params['step']]])
    
    out_dir_train = os.path.join(out_dir_meta,'bigger_ck_'+str(split_num)+'_'+strs_append)
    print out_dir_train

    train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
    test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
    mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
    std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'

    data_transforms = {}
    data_transforms['train']= transforms.Compose([
        lambda x: augmenters.random_crop(x,32),
        lambda x: augmenters.horizontal_flip(x),
        transforms.ToTensor(),
        lambda x: x*255.
    ])
    data_transforms['val']= transforms.Compose([
        lambda x: augmenters.crop_center(x,32,32),
        transforms.ToTensor(),
        lambda x: x*255.
        ])

    train_data = dataset.CK_48_Dataset(train_file, mean_file, std_file, data_transforms['train'])
    test_data = dataset.CK_48_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    
    
    batch_size = 16
    batch_size_val = 16


    util.makedirs(out_dir_train)
    
    train_params = dict(out_dir_train = out_dir_train,
                train_data = train_data,
                test_data = test_data,
                batch_size = batch_size,
                batch_size_val = batch_size_val,
                num_epochs = num_epochs,
                save_after = 1,
                disp_after = 1,
                plot_after = 10,
                test_after = 1,
                lr = lr,
                dec_after = dec_after, 
                model_name = 'pytorch_mat_capsules',
                criterion = 'spread',
                gpu_id = 1,
                num_workers = 0,
                model_file = None,
                epoch_start = 0,
                margin_params = margin_params,
                network_params = dict(A=32,B=32,C=32,D=32,E=32,r=1))

    train_model(**train_params)

 
def main():
    
    
    out_dir_meta = '../experiments/dynamic_capsules/'
    num_epochs = 108
    dec_after = ['exp',0.96,50,1e-6]
    lr = 0.001
    split_num = 0
    im_size = 28
    # margin_params = {'step':1,'start':0.2}

    strs_append = '_'.join([str(val) for val in [num_epochs,dec_after[0],lr]])
    
    out_dir_train = os.path.join(out_dir_meta,'ck_bigger_'+str(split_num)+'_'+strs_append)
    print out_dir_train

    train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
    test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
    mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
    std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'

    data_transforms = {}
    data_transforms['train']= transforms.Compose([
        # lambda x: augmenters.random_crop(x,32),
        lambda x: augmenters.horizontal_flip(x),
        transforms.ToTensor(),
        lambda x: x*255.
    ])
    data_transforms['val']= transforms.Compose([
        # lambda x: augmenters.crop_center(x,32,32),
        transforms.ToTensor(),
        lambda x: x*255.
        ])

    train_data = dataset.CK_RS_Dataset(train_file, mean_file, std_file, im_size, data_transforms['train'])
    test_data = dataset.CK_RS_Dataset(test_file, mean_file, std_file, im_size, data_transforms['val'])
    # train_data = dataset.CK_96_Dataset(train_file, mean_file, std_file, data_transforms['train'])
    # test_data = dataset.CK_96_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    
    network_params = dict(n_classes=8,
                        conv_layers = None,
                        # [[259,9,1]],
                        caps_layers = None,
                        # [[32,8,5,2],[32,8,5,2],[32,8,5,2],[8,16,6,1]],
                        r=3)
    
    batch_size = 128
    batch_size_val = 128


    util.makedirs(out_dir_train)
    
    train_params = dict(out_dir_train = out_dir_train,
                train_data = train_data,
                test_data = test_data,
                batch_size = batch_size,
                batch_size_val = batch_size_val,
                num_epochs = num_epochs,
                save_after = 1,
                disp_after = 1,
                plot_after = 10,
                test_after = 1,
                lr = lr,
                dec_after = dec_after, 
                model_name = 'dynamic_capsules',
                criterion = 'margin',
                gpu_id = 1,
                num_workers = 0,
                model_file = None,
                epoch_start = 0,
                network_params = network_params)

    print train_params
    param_file = os.path.join(out_dir_train,'params.txt')
    all_lines = []
    for k in train_params.keys():
        str_print = '%s: %s' % (k,train_params[k])
        print str_print
        all_lines.append(str_print)
    util.writeFile(param_file,all_lines)

    train_model(**train_params)




if __name__=='__main__':
    main()