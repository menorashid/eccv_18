from torchvision import models
import torch.nn as nn

import numpy as np
import scipy.misc
import torch
from CapsuleLayer import CapsuleLayer
from dynamic_capsules import Dynamic_Capsule_Model_Super
from torch.autograd import Variable
import torch.nn.functional as F
import math
from vgg_capsule_disfa_withrecon import Vgg_Capsule_Disfa_Recon

class Vgg_Capsule_Disfa_Bigprimary_Recon(Vgg_Capsule_Disfa_Recon):

    def __init__(self,n_classes,loss,in_size = 224, r=3,):
        super(Dynamic_Capsule_Model_Super, self).__init__()
        self.num_classes = n_classes
        self.in_size = in_size
        self.vgg_base = torch.load('models/pytorch_vgg_face_just_conv.pth')
        # print self.vgg_base

        self.reconstruct = True
        self.class_loss = loss
        
        self.features = []
        
        self.features.append(CapsuleLayer(64, 1, 512, 8, kernel_size=3, stride=2, num_iterations=r))
        
        self.features.append(CapsuleLayer(n_classes, 64, 8, 16, kernel_size=6, stride=1, num_iterations=r))
        
        self.features = nn.Sequential(*self.features)
        self.reconstruction_loss = nn.MSELoss(size_average=True)
        self.decoder = nn.Sequential(
            nn.Linear(16 * self.num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024)
        )
        self.upsampler = nn.Upsample(size=(self.in_size,self.in_size), mode='bilinear')


class Network:
    def __init__(self,n_classes,loss,in_size,r, init=False):
        # print 'BN',bn
        model = Vgg_Capsule_Disfa_Bigprimary_Recon(n_classes,loss,in_size,r)
        
        if init:
            for idx_m,m in enumerate(model.features):
                if isinstance(m, CapsuleLayer):
                    
                    if m.num_in_capsules==1:
                        nn.init.xavier_normal(m.capsules.weight.data)
                        nn.init.constant(m.capsules.bias.data,0.)
                    else:
                        nn.init.normal(m.route_weights.data, mean=0, std=0.1)
                elif isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                    print m
                    nn.init.xavier_normal(m.weight.data)
                    nn.init.constant(m.bias.data,0.)
                
        self.model = model
        
    
    def get_lr_list(self, lr):
        lr_list =[]
        for lr_curr,param_set in zip(lr,[self.model.vgg_base,self.model.features,self.model.decoder]):
            if lr_curr==0:
                for param in param_set.parameters():
                    param.requires_grad = False
            else:
                lr_list.append({'params': param_set.parameters(), 'lr': lr_curr})

        # lr_list= [{'params': self.model.vgg_base.parameters(), 'lr': lr[0]}] +\
        #         [{'params': self.model.features.parameters(), 'lr': lr[1]}]
        return lr_list

def main():
    import numpy as np
    import torch
    from torch.autograd import Variable
    import torch.optim as optim

    n_classes = 10
    loss = nn.CrossEntropyLoss()
    r = 1
    in_size = 224

    net = Network(n_classes= n_classes, loss = loss, in_size= in_size,r= r, init = False)
    print net.model
    labels = np.random.randn(16,n_classes)
    labels[labels>0.5]=1
    labels[labels<0.5]=0
    labels = np.zeros(16)

    net.model = net.model.cuda()
    print net.model
    input = np.random.randn(16,3,224,224)
    
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    optimizer = optim.Adam(net.model.parameters(),lr=0.00005)
    labels = Variable(torch.LongTensor(labels).cuda())
    # output = net.model(input)
    # print output.data.shape
    
    # criterion(output,labels)
    epochs = 1000
    for epoch in range(epochs):
        # inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
        # labelsv = Variable(torch.FloatTensor(labels[i])).view(1, -1)
        # print labelsv
        output = net.model(input)
        loss = net.model.margin_loss(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # losses.append(loss.data.mean())
        print('[%d/%d] Loss: %.3f' % (epoch+1, epochs, loss.data))
    print output
    print labels
        

    # criterion = Spread_Loss(50,5)
    # for epoch_num in range(53):
    #     print epoch_num,criterion(output,labels,epoch_num)



if __name__=='__main__':
    main()



