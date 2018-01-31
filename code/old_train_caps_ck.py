# -*- coding: utf-8 -*-

'''
The Capsules Network.

@author: Yuxian Meng
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
# import models

from models.matrix_capsules import PrimaryCaps, ConvCaps

from train_ck import augment_image
import scipy.misc
import numpy as np
import dataset
from torchvision import transforms

class CapsNet(nn.Module):
    def __init__(self,A=32,B=32,C=32,D=32, E=10,r = 3):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=2)
        self.primary_caps = PrimaryCaps(A,B)
        self.convcaps1 = ConvCaps(B, C, kernel = 3, stride=2,iteration=r,
                                  coordinate_add=False, transform_share = False)
        self.convcaps2 = ConvCaps(C, D, kernel = 3, stride=1,iteration=r,
                                  coordinate_add=False, transform_share = False)
        self.classcaps = ConvCaps(D, E, kernel = 0, stride=1,iteration=r,
                                  coordinate_add=True, transform_share = True) 
        
        
    def forward(self,x,lambda_): #b,1,28,28
        # print 'input',x.size()
        x = F.relu(self.conv1(x)) #b,32,12,12
        # print 'input',x.size()
        x = self.primary_caps(x) #b,32*(4*4+1),12,12
        # print 'input',x.size()
        x = self.convcaps1(x,lambda_) #b,32*(4*4+1),5,5
        # print 'input',x.size()
        x = self.convcaps2(x,lambda_) #b,32*(4*4+1),3,3
        # print 'input',x.size()
        x = self.classcaps(x,lambda_).view(-1,10*16+10) #b,10*16+10
        # print 'input',x.size()
        return x
    
    def loss(self, x, target, m): #x:b,10 target:b
        b = x.size(0)
        a_t = torch.cat([x[i][target[i]] for i in range(b)]) #b
        a_t_stack = a_t.view(b,1).expand(b,10).contiguous() #b,10
        u = m-(a_t_stack-x) #b,10
        mask = u.ge(0).float() #max(u,0) #b,10
        loss = ((mask*u)**2).sum()/b - m**2  #float
        return loss
    
    def loss2(self,x ,target):
        loss = F.cross_entropy(x,target)
        return loss

class CapsNet_ck(nn.Module):
    def __init__(self,A=32,B=32,C=32,CC = 32, D=32, E=10,r = 3):
        super(CapsNet_ck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=2)
        # self.conv2 = nn.Conv2d(in_channels=A, out_channels=A,
        #                        kernel_size=5, stride=2)
        
        self.primary_caps = PrimaryCaps(A,B)
        self.convcaps1 = ConvCaps(B, C, kernel = 3, stride=2,iteration=r,
                                  coordinate_add=False, transform_share = False)
        self.convcaps2 = ConvCaps(C, CC, kernel = 3, stride=2,iteration=r,
                                  coordinate_add=False, transform_share = False)
        
        self.convcaps3 = ConvCaps(C, D, kernel = 3, stride=2,iteration=r,
                                  coordinate_add=False, transform_share = False)
        self.classcaps = ConvCaps(D, E, kernel = 0, stride=1,iteration=r,
                                  coordinate_add=True, transform_share = True) 
        self.E = E
        
    def forward(self,x,lambda_): #b,1,28,28
        # print 'input',x.size()
        x = F.relu(self.conv1(x)) #b,32,12,12
        # print 'conv1',x.size()
        x = self.primary_caps(x) #b,32*(4*4+1),12,12
        # print 'primarycaps',x.size()
        x = self.convcaps1(x,lambda_) #b,32*(4*4+1),5,5
        # print 'convcaps1',x.size()
        x = self.convcaps2(x,lambda_) #b,32*(4*4+1),3,3
        # print 'convcaps2',x.size()
        x = self.convcaps3(x,lambda_) #b,32*(4*4+1),3,3
        # print 'convcaps3',x.size()
        
        x = self.classcaps(x,lambda_).view(-1,self.E*16+self.E) #b,10*16+10
        # print x.size()

        return x
    
    def loss(self, x, target, m): #x:b,10 target:b
        b = x.size(0)
        a_t = torch.cat([x[i][target[i]] for i in range(b)]) #b
        a_t_stack = a_t.view(b,1).expand(b,10).contiguous() #b,10
        u = m-(a_t_stack-x) #b,10
        mask = u.ge(0).float() #max(u,0) #b,10
        loss = ((mask*u)**2).sum()/b - m**2  #float
        return loss
    
    def loss2(self,x ,target):
        loss = F.cross_entropy(x,target)
        return loss


def main():
    from models.utils import get_args, get_dataloader

    # args = get_args()
    # print args

    split_num = 0
        
    train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
    test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
    mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
    std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'

    list_of_to_dos = ['flip','rotate']
    mean_im = scipy.misc.imresize(scipy.misc.imread(mean_file),(48,48)).astype(np.float32)
    std_im = scipy.misc.imresize(scipy.misc.imread(std_file),(48,48)).astype(np.float32)
    
    batch_size = 32
    clip = 5
    disable_cuda = False
    gpu = 2
    lr = 0.02
    num_epochs = 5
    disp_after = 1
    r = 3
    use_cuda = True

    batch_size_val = 64
    save_after = 1
    test_after = 1
    
    plot_after = 10

    data_transforms = {}

    data_transforms['train']= transforms.Compose([
        lambda x: augment_image(x, list_of_to_dos, mean_im = mean_im, std_im = std_im,im_size = 48),
        transforms.ToTensor(),
        lambda x: x*255.
    ])

    data_transforms['val']= transforms.Compose([
        transforms.ToTensor(),
        lambda x: x*255.
        ])

    
    # train_loader, test_loader = get_dataloader(batch_size)
    # for data in train_loader:
    #     imgs,labels = data
    #     print labels
    #     break
    # return
    
    our_data = True
    train_data = dataset.CK_48_Dataset(train_file, mean_file, std_file, data_transforms['train'])
    test_data = dataset.CK_48_Dataset(test_file, mean_file, std_file, data_transforms['val'])
    
    train_loader = torch.utils.data.DataLoader(train_data, 
                        batch_size=batch_size,
                        shuffle=True, 
                        num_workers=0)
    
    test_loader = torch.utils.data.DataLoader(test_data, 
                        batch_size=batch_size_val,
                        shuffle=False, 
                        num_workers=0)

    # -batch_size=64 -lr=2e-2 -num_epochs=5 -r=1 -print_freq=5
    steps = len(train_loader.dataset)//batch_size
    print 'steps'
    lambda_ = 1e-2 #TODO:find a good schedule to increase lambda and m
    m = 0.2
    
    # A,B,C,D,E,r = 64,8,16,16,10,r # a small CapsNet
    # model = CapsNet(A,B,C,D,E,r)

    A,B,C,CC,D,E,r = 32,8,16,16,16,8,r # additional conv-caps layer for bigger input

    # # A,B,C,CC,D,E,r = 64,8,16,16,16,8,r #  additional conv-caps layer for bigger input
    model = CapsNet_ck(A,B,C,CC,D,E,r)

    # print model


    with torch.cuda.device(gpu):
#        print(gpu, type(gpu))
        # if pretrained:
        #     model.load_state_dict(torch.load(pretrained))
        #     m = 0.8
        #     lambda_ = 0.9
        if use_cuda:
            print("activating cuda")
            model.cuda()
            
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience = 1)
        for epoch in range(num_epochs):
            print 'm',m
            #Train
            print("Epoch {}".format(epoch))
            b = 0
            correct = 0
            for data in train_loader:
                b += 1
                # if lambda_ < 1:
                #     lambda_ += 2e-1/steps
                if m < 0.9:
                    m += 2e-1/steps
                optimizer.zero_grad()

                if our_data:
                    imgs = data['image']
                    labels = data['label']
                    
                else:
                    imgs,labels = data #b,1,28,28; #b

                imgs,labels = Variable(imgs),Variable(labels)
                if use_cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()

                raw_input()
                out = model(imgs,lambda_) #b,10,17
                out_poses, out_labels = out[:,:-10],out[:,-10:] #b,16*10; b,10
                loss = model.loss(out_labels, labels, m)
                # loss = model.loss2(out_labels,labels)
                # torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                loss.backward()
                optimizer.step()
                #stats
                pred = out_labels.max(1)[1] #b
                acc = pred.eq(labels).cpu().sum().data[0]
                correct += acc
                if b % disp_after == 0:                          
                    print("batch:{}, loss:{:.4f}, acc:{:}/{}".format(
                            b, loss.data[0],acc, batch_size))
            #     break

            # break

            acc = correct/float(len(train_loader.dataset))
            print("Epoch{} Train acc:{:4}".format(epoch, acc))
            scheduler.step(acc)
            if epoch%save_after == 0:
                torch.save(model.state_dict(), "./model_{}.pth".format(epoch))
            
            #Test
            if epoch%test_after ==0:
                print('Testing...')
                correct = 0
                for data in test_loader:
                    if our_data:
                        imgs = data['image']
                        labels = data['label']
                    else:
                        imgs,labels = data #b,1,28,28; #b
                    imgs,labels = Variable(imgs),Variable(labels)
                    if use_cuda:
                        imgs = imgs.cuda()
                        labels = labels.cuda()
                    out = model(imgs,lambda_) #b,10,17
                    out_poses, out_labels = out[:,:-10],out[:,-10:] #b,16*10; b,10
                    # loss = model.loss(out_labels, labels, m)
                    loss = model.loss2(out_labels, labels)
                    #stats
                    pred = out_labels.max(1)[1] #b
                    acc = pred.eq(labels).cpu().sum().data[0]
                    correct += acc
                acc = correct/float(len(test_loader.dataset))
                print("Epoch{} Test acc:{:4}".format(epoch, acc))



if __name__ == "__main__":
    main()
            
            
            
            

        
        
