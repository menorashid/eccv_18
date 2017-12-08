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
                epoch_start = 0):

    util.mkdir(out_dir_train)
    log_file = os.path.join(out_dir_train,'log.txt')
    plot_file = os.path.join(out_dir_train,'loss.jpg')
    log_arr = []
    plot_arr = [[],[]]
    plot_val_arr = [[],[]]

    network = models.get(model_name)
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
    
    optimizer = optim.SGD(network.get_lr_list(lr), lr=0, momentum=0.9)

    if dec_after is not None:
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=dec_after, gamma=0.1)

    for num_epoch in range(epoch_start,num_epochs):

        for num_iter_train,batch in enumerate(train_dataloader):
            
            data = Variable(batch['image'].cuda())
            labels = Variable(torch.LongTensor(batch['label']).cuda())
            optimizer.zero_grad()
            loss = criterion(model(data), labels)    
            loss_iter = loss.data[0]
            loss.backward()
            optimizer.step()
            
            
            num_iter = num_epoch*len(train_dataloader)+num_iter_train
            plot_arr[0].append(num_iter); plot_arr[1].append(loss_iter)

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

        if dec_after is not None:
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
    
        




if __name__=='__main__':
    main()