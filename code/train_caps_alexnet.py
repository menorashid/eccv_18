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
import dataset
import glob
import sklearn.metrics



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
                lr = 0.0001,dec_after = 100, model_name = 'alexnet'):

    util.mkdir(out_dir_train)
    log_file = os.path.join(out_dir_train,'log.txt')
    plot_file = os.path.join(out_dir_train,'loss.jpg')
    log_arr = []
    plot_arr = [[],[]]
    plot_val_arr = [[],[]]

    train_data = dataset.Horse_Image_Dataset(train_file,data_transforms['train'])
    test_data = dataset.Horse_Image_Dataset(test_file,data_transforms['val'])
    
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
                        num_workers=0)

        

    class_weights = get_class_weights(util.readLinesFromFile(train_file))
    
    torch.cuda.device(0)
    iter_begin = 0

    network = models.get('caps_alexnet_simple')

    model = network.model.cuda()
    # model.train(True)
    # criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor(class_weights).cuda())
    
    optimizer = optim.Adam(network.get_lr_list(lr), lr=0)

    if dec_after is not None:
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=dec_after, gamma=0.1)

    for num_epoch in range(num_epochs):

        for num_iter_train,batch in enumerate(train_dataloader):
            
            # print batch['image'].shape,torch.min(batch['image']),torch.max(batch['image'])
            # im = np.transpose(batch['image'][0].numpy(),(1,2,0))
            # im = batch['image'][0].numpy()
                
            # print im.shape
            # scipy.misc.imsave('../scratch/check.jpg',im)
            # raw_input()
            data = Variable(batch['image'].cuda())
            one_hot =  models.utils.one_hot_encode(batch['label'],2)
            loss_weights = torch.FloatTensor(np.tile(np.array(class_weights)[np.newaxis,:],(one_hot.shape[0],1)))
            one_hot = torch.mul(one_hot,loss_weights)
            labels = Variable(one_hot).cuda()
            
            # labels = Variable(models.utils.one_hot_encode(batch['label'],2)).cuda()
            output = model(data) # output from DigitCaps (out_digit_caps)
            loss = model.loss(data, output, labels) # pass in data for image reconstruction
            loss.backward()
            loss_iter = loss.data[0]
            optimizer.step()
            
            num_iter = num_epoch*len(train_dataloader)+num_iter_train
            # num_iter +=1
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
            for num_iter_test,batch in enumerate(test_dataloader):
                # data = Variable(batch['image'].cuda())
                # labels = Variable(torch.LongTensor(batch['label']).cuda())
                # loss = criterion(model(data), labels)    
                # loss_iter = loss.data[0]

                data = Variable(batch['image'].cuda())
                # labels = Variable(models.utils.one_hot_encode(batch['label'],2)).cuda()

                one_hot =  models.utils.one_hot_encode(batch['label'],2)
                loss_weights = torch.FloatTensor(np.tile(np.array(class_weights)[np.newaxis,:],(one_hot.shape[0],1)))
                one_hot = torch.mul(one_hot,loss_weights)
                labels = Variable(one_hot).cuda()
                

                output = model(data) # output from DigitCaps (out_digit_caps)
                loss = model.loss(data, output, labels) # pass in data for image reconstruction
                # loss.backward()
                loss_iter = loss.data[0]
                optimizer.step()

                # test_epoch = num_epoch/test_after
                num_iter = num_epoch*len(train_dataloader)+num_iter_test
                # +=1 
                # 
                plot_val_arr[0].append(num_iter); plot_val_arr[1].append(loss_iter)

                str_display = 'lr: %.6f, val iter: %d, val loss: %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter)
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
    
    print plot_arr[0]

    util.writeFile(log_file, log_arr)
    if len(plot_val_arr[0])==0:
        visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
    else:
        visualize.plotSimple([(plot_arr[0],plot_arr[1]),(plot_val_arr[0],plot_val_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train','Val'])   

            

def test_model(out_dir_train,
                model_num,
                train_file,
                test_file,
                data_transforms,
                batch_size_val =None):

    out_dir_results = os.path.join(out_dir_train,'results_model_'+str(model_num))
    util.mkdir(out_dir_results)
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    log_file = os.path.join(out_dir_results,'log.txt')
    log_arr=[]

    test_data = dataset.Horse_Image_Dataset(test_file,data_transforms['val'])
    
    if batch_size_val is None:
        batch_size_val = len(test_data)
    
    class_weights = get_class_weights(util.readLinesFromFile(train_file))
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_val,
                        shuffle=False, num_workers=1)

    torch.cuda.device(0)
    iter_begin = 0
    model = torch.load(model_file)
    model.cuda()
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    predictions = []
    labels_all = []
    out_all = []

    for num_iter,batch in enumerate(test_dataloader):
                
        # batch = test_dataloader.next() 
        labels_all.append(batch['label'].numpy())
        

        data = Variable(batch['image'].cuda())
        one_hot =  models.utils.one_hot_encode(batch['label'],2)
        loss_weights = torch.FloatTensor(np.tile(np.array(class_weights)[np.newaxis,:],(one_hot.shape[0],1)))
        one_hot = torch.mul(one_hot,loss_weights)
        
        labels = Variable(one_hot).cuda()
        output = model(data) # output from DigitCaps (out_digit_caps)

        v_magnitud = torch.sqrt((output**2).sum(dim=2, keepdim=True))
        out = v_magnitud.data.cpu().numpy().squeeze()
        out_all.append(out)
        # raw_input()


        predictions_curr = np.argmax(out,1)
        # v_magnitud.data.max(1, keepdim=True)[1].cpu().numpy().squeeze()
        predictions.append(predictions_curr)
        
        loss = model.loss(data, output, labels) # pass in data for image reconstruction
        loss_iter = loss.data[0]
                
        # output = model(data)
        # predictions.append(np.argmax(output.data.cpu().numpy(),1))
    
        # loss = criterion(output, labels)    
        # loss_iter = loss.data[0]

        str_display = 'iter: %d, val loss: %.4f' %(num_iter,loss_iter)
        log_arr.append(str_display)
        print str_display
        

        util.writeFile(log_file, log_arr)
        
    out_all = np.concatenate(out_all,0)
    predictions = np.concatenate(predictions)
    labels_all = np.concatenate(labels_all)
    y_true = np.zeros((labels_all.shape[0],2))
    y_true[labels_all==0,0]=1
    y_true[labels_all==1,1]=1

    f1 = sklearn.metrics.f1_score(labels_all, predictions)
    ap = sklearn.metrics.average_precision_score(y_true, out_all)
    roc_auc = sklearn.metrics.roc_auc_score(y_true, out_all, average='macro')
    str_display = 'f1: %.4f' %(f1)
    print str_display
    log_arr.append(str_display)
    
    str_display = 'ap: %.4f' %(ap)
    print str_display
    log_arr.append(str_display)
    
    str_display = 'roc_auc: %.4f' %(roc_auc)
    print str_display
    log_arr.append(str_display)
    
    util.writeFile(log_file, log_arr)


def ft_caps_alexnet_simple(pre_str=None):


    data_dir_meta = '../data/horse_51'
    # split_dir = os.path.join(data_dir_meta,'train_test_split')
    # num_splits = 5

    split_dir = os.path.join(data_dir_meta,'train_test_split_horse_based')
    num_splits = 6
    # visualize.writeHTMLForFolder(data_dir_meta)
    # return

    num_epochs = 30
    save_after = 5
    disp_after = 1
    plot_after = 10
    test_after = 1
    # lr = [0.0001, 0.001,0.01]
    lr = [0, 0.001,0.01]
    dec_after = 30
    batch_size = None
    batch_size_val = None
    num_epochs_test = num_epochs-1
    # num_epochs_test = 25

    if pre_str is None:
        out_dir_train_meta = '../experiments/caps_alexnet_simple_'+'_'.join([str(val) for val in [os.path.split(split_dir)[1],num_epochs,dec_after]+lr])
    else:
        out_dir_train_meta = '../experiments/'+'_'.join([str(val) for val in [pre_str,os.path.split(split_dir)[1],num_epochs,dec_after]+lr])
    util.mkdir(out_dir_train_meta)

    for split_num in range(num_splits):
        out_dir_train = os.path.join(out_dir_train_meta,'split_'+str(split_num))
        train_file = os.path.join(split_dir,'train_'+str(split_num)+'.txt')


        val_file = os.path.join(split_dir,'test_'+str(split_num)+'.txt')
        test_file = os.path.join(split_dir,'test_'+str(split_num)+'.txt')

        # class_weights = get_class_weights(util.readLinesFromFile(train_file))

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

        # train_model(out_dir_train,
        #             train_file,
        #             test_file,
        #             data_transforms,
        #             num_epochs = num_epochs,
        #             save_after = save_after,
        #             disp_after = disp_after,
        #             plot_after = plot_after,
        #             test_after = test_after,
        #             lr = lr,
        #             batch_size = batch_size,
        #             batch_size_val = batch_size_val,
        #             dec_after = dec_after)

        test_model(out_dir_train,
                    num_epochs_test,
                    train_file,
                    test_file,
                    data_transforms,
                    batch_size_val = batch_size_val
                    )
        # break
        
def main():
    # ft_horse_alexnet()
    pre_pre_str = 'caps_alexnet_simple'
    print pre_pre_str
    # pre_pre_str = 'alexnet_'
    # for rep in range(5):
    #     print 'REP NO',rep
    #     ft_caps_alexnet_simple(pre_pre_str+str(rep))
        
    # return 
    meta_exp_dir = '../experiments'
    # all_exp_dirs = glob.glob(os.path.join(meta_exp_dir,'*'))
    # all_exp_dirs = all_exp_dirs+glob.glob(os.path.join(meta_exp_dir,'horse_based','*'))
    # all_dirs = ha_exp_dirs+a_exp_dirs
    # all_exp_dirs = [os.path.join(meta_exp_dir,str_curr+'_train_test_split_horse_based_20_20_0_1e-05_0.0001') for str_curr in ['alexnet','horse_alexnet']]
    all_exp_dirs = glob.glob(os.path.join(meta_exp_dir,pre_pre_str+'*_train_test_split_horse_based_30_30_0_*'))

    for model_num in [29]:
        rep_accuracy = []
        for dir_curr in all_exp_dirs:
            accuracy =[]
            splits = list(glob.glob(os.path.join(dir_curr,'split_*')))
            if len(splits)<6:
                continue

            for split_dir in glob.glob(os.path.join(dir_curr,'split_*')):
                # print split_dir
                result_dirs = glob.glob(os.path.join(split_dir,'results_model_'+str(model_num)))
                # if len(result_dirs)==0:
                #     continue
                # result_dirs.sort()
                # result_dirs = [res_dir_curr for res_dir_curr in result_dirs if os.path.exists(os.path.join(res_dir_curr,'log.txt'))]
                # if len(result_dirs)>1:
                #     result_dir = result_dirs[0]
                # else: 
                result_dir = result_dirs[-1]
                lines = util.readLinesFromFile(os.path.join(result_dir,'log.txt'))
                accuracy_curr = [float(line_curr.split(' ')[-1]) for line_curr in lines[-3:]]
                # print accuracy_curr
                accuracy.append(accuracy_curr)
            # rep_accuracy.append(np.mean(np.array(accuracy),0))  
            rep_accuracy.extend(accuracy)
        rep_accuracy = np.array(rep_accuracy)

        print rep_accuracy.shape
        print model_num
        print np.mean(rep_accuracy,0)
        print np.std(rep_accuracy,0)



        




if __name__=='__main__':
    main()