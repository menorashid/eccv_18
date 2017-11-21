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


def get_new_batch(train_files_iterator, data_transformer, batch_size):
    all_data = []
    labels = []
    for im_num in range(batch_size): 
        train_file_curr = train_files_iterator.next()
        train_file_curr,label = train_file_curr.split(' ')
        label = int(label)
        data_curr = data_transformer(Image.open(train_file_curr))
        # Image.open(train_file_curr)
        # data_transformer(Image.open(train_file_curr))
        # print data_curr.shape,torch.min(data_curr), torch.max(data_curr)
        # data_curr = scipy.misc.imresize(data_curr,(im_size,im_size))

        all_data.append(data_curr)
        labels.append(label)

    all_data = torch.stack(all_data,dim=0)
    labels = np.array(labels,dtype=int)
    return all_data,labels

def get_class_weights(train_files):
    classes = [int(line_curr.split(' ')[1]) for line_curr in train_files]
    counts = np.array([classes.count(0),classes.count(1)])
    print counts
    counts = counts/float(np.sum(counts))
    counts = 1./counts
    counts = counts/float(np.sum(counts))
    print counts
    return counts


def train_model(out_dir_train,
                train_file,
                test_file,
                data_transforms,
                batch_size = None,
                batch_size_val =None,
                num_epochs = 100,
                save_after = 20,
                disp_after = 1,
                plot_after = 10,
                test_after = 1,
                lr = 0.0001,dec_after = 100):

    util.mkdir(out_dir_train)
    log_file = os.path.join(out_dir_train,'log.txt')
    plot_file = os.path.join(out_dir_train,'loss.jpg')
    log_arr = []
    plot_arr = [[],[]]
    plot_val_arr = [[],[]]

    

    train_files = util.readLinesFromFile(train_file)
    test_files = util.readLinesFromFile(test_file)
    print len(test_files)
    
    if batch_size is None:
        batch_size = len(train_files)

    if batch_size_val is None:
        batch_size_val = len(test_files)


    class_weights = get_class_weights(train_files)
    # print class_weights
    
    epoch_size = len(train_files)/batch_size
    num_iterations = epoch_size*num_epochs
    save_after = save_after*epoch_size
    test_after = test_after* epoch_size



    print 'epoch_size',epoch_size, 'num_iterations', num_iterations, 'batch_size',batch_size, 'batch_size_val',batch_size_val

    torch.cuda.device(0)
    iter_begin = 0

    network = models.get('alexnet')

    model = network.model.cuda()
    print model
    print network.get_lr_list(lr)
    # get_alexnet().cuda()
    model.train(True)
    criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor(class_weights).cuda())
    
    optimizer = optim.SGD(network.get_lr_list(lr)
            , lr=0, momentum=0.9)

    if dec_after is not None:
        dec_after = dec_after* epoch_size
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=dec_after, gamma=0.1)

    random.shuffle(train_files)
    train_files_iterator = itertools.cycle(train_files)    
    test_files_iterator = itertools.cycle(test_files)    
    

    for num_iter in range(iter_begin,num_iterations):
        if dec_after is not None:
            exp_lr_scheduler.step()
        
        data, labels = get_new_batch(train_files_iterator, data_transforms['train'],batch_size)        
        data = Variable(data.cuda())
        labels = Variable(torch.LongTensor(labels).cuda())
        optimizer.zero_grad()
        loss = criterion(model(data), labels)    
        loss_iter = loss.data[0]
        loss.backward()
        optimizer.step()
        
        plot_arr[0].append(num_iter); plot_arr[1].append(loss_iter)
        str_display = 'lr: %.6f, iter: %d, loss: %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter)
        log_arr.append(str_display)
        print str_display

        if num_iter % test_after == 0 :
            model.eval()
            data, labels = get_new_batch(test_files_iterator, data_transforms['val'],batch_size_val)        
            # print data.shape
            data = Variable(data.cuda())
            labels = Variable(torch.LongTensor(labels).cuda())
            loss = criterion(model(data), labels)    
            loss_iter = loss.data[0]
            plot_val_arr[0].append(num_iter); plot_val_arr[1].append(loss_iter)
            str_display = 'lr: %.6f, iter: %d, val loss: %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter)
            log_arr.append(str_display)
            print str_display
            model.train(True)
            

        if num_iter % epoch_size == 0 and num_iter>0:
            print 'shuffling'
            random.shuffle(train_files)
            train_files_iterator = itertools.cycle(train_files)    

        if (num_iter % plot_after== 0 and num_iter>0) or num_iter==num_iterations-1:
            util.writeFile(log_file, log_arr)
            if len(plot_val_arr[0])==0:
                visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
            else:
                visualize.plotSimple([(plot_arr[0],plot_arr[1]),(plot_val_arr[0],plot_val_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train','Val'])

        if num_iter % save_after == 0 or num_iter==num_iterations-1:
            out_file = os.path.join(out_dir_train,'model_'+str(num_iter)+'.pt')
            print 'saving',out_file
            torch.save(model,out_file)
            

def test_model(out_dir_train,
                model_num,
                train_file,
                test_file,
                data_transforms,
                batch_size_val =None,):

    out_dir_results = os.path.join(out_dir_train,'results_model_'+str(model_num))
    util.mkdir(out_dir_results)
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    log_file = os.path.join(out_dir_results,'log.txt')
    log_arr=[]

    test_files = util.readLinesFromFile(test_file)
    print len(test_files)
    
    if batch_size_val is None:
        batch_size_val = len(test_files)
    
    epoch_size = len(test_files)/batch_size_val
    num_iterations = epoch_size


    print 'epoch_size',epoch_size, 'num_iterations', num_iterations

    torch.cuda.device(0)
    iter_begin = 0
    model = torch.load(model_file)
    model.cuda()
    model.eval()
    
    # train_files = util.readLinesFromFile(train_file)
    # class_weights = get_class_weights(test_files)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.classifier[-1].parameters(), lr=lr, momentum=0.9)

    test_files_iterator = itertools.cycle(test_files)    
    
    predictions = []
    labels_all = []

    for num_iter in range(iter_begin,num_iterations):
        
        data, labels = get_new_batch(test_files_iterator, data_transforms['val'],batch_size_val)        
        labels_all.append(labels)

        data = Variable(data.cuda())
        labels = Variable(torch.LongTensor(labels).cuda())
        output = model(data)
        predictions.append(np.argmax(output.data.cpu().numpy(),1))
    
        loss = criterion(output, labels)    
        loss_iter = loss.data[0]
        # plot_val_arr[0].append(num_iter); plot_val_arr[1].append(loss_iter)
        str_display = 'iter: %d, val loss: %.4f' %(num_iter,loss_iter)
        log_arr.append(str_display)
        print str_display
        

        util.writeFile(log_file, log_arr)
        
    predictions = np.concatenate(predictions)[:len(test_files)]
    labels_all = np.concatenate(labels_all)[:len(test_files)]
    
    accuracy = np.sum(predictions==labels_all)/float(labels_all.size)
    str_display = 'accuracy: %.4f' %(accuracy)
    print str_display
    log_arr.append(str_display)
    
    util.writeFile(log_file, log_arr)
    

def main():


    data_dir_meta = '../data/horse_51'
    split_dir = os.path.join(data_dir_meta,'train_test_split')
    num_splits = 5

    # split_dir = os.path.join(data_dir_meta,'train_test_split_horse_based')
    # num_splits = 6
    # visualize.writeHTMLForFolder(data_dir_meta)
    # return

    num_epochs = 20
    save_after = 20
    disp_after = 1
    plot_after = 10
    test_after = 1
    lr = [0, 0.00001,0.0001]
    dec_after = 50

    out_dir_train_meta = '../experiments/alexnet_'+'_'.join([str(val) for val in [num_epochs,dec_after]+lr])
    util.mkdir(out_dir_train_meta)

    for split_num in range(num_splits):
        out_dir_train = os.path.join(out_dir_train_meta,'split_'+str(split_num))
        train_file = os.path.join(split_dir,'train_'+str(split_num)+'.txt')
        val_file = os.path.join(split_dir,'test_'+str(split_num)+'.txt')
        test_file = os.path.join(split_dir,'test_'+str(split_num)+'.txt')

        data_transforms = {
            'train': transforms.Compose([
                transforms.Scale([256,256]),
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        

        train_model(out_dir_train,
                    train_file,
                    test_file,
                    data_transforms,
                    num_epochs = num_epochs,
                    save_after = save_after,
                    disp_after = disp_after,
                    plot_after = plot_after,
                    test_after = test_after,
                    lr = lr,
                    dec_after = dec_after)

        test_model(out_dir_train,
                    num_epochs-1,
                    train_file,
                    test_file,
                    data_transforms)
        # break

    


if __name__=='__main__':
    main()