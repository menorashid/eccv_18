from torchvision import models
import torch.nn as nn

import numpy as np
import scipy.misc
import torch
from CapsuleLayer import CapsuleLayer
from dynamic_capsules import Dynamic_Capsule_Model_Super
import math
import torch.nn.functional as F
from torch.autograd import Variable
from spread_loss import Spread_Loss

class Khorrami_Capsule_Reconstruct(Dynamic_Capsule_Model_Super):

    def __init__(self,n_classes,spread_loss_params,pool_type='max',r=3,reconstruct= False,class_weights=None):
        super(Dynamic_Capsule_Model_Super, self).__init__()
        
        self.reconstruct = reconstruct
        self.num_classes = n_classes
        self.in_size = 96

        # spread_loss_params = dict(init_margin= spread_loss_params['init_margin'],
        #                          decay_steps = spread_loss_params['decay_steps'],
        #                          end_epoch = spread_loss_params['end_epoch'],
        #                          max_margin = spread_loss_params['max_margin'])
        # print spread_loss_params
        self.spread_loss = Spread_Loss(**spread_loss_params)
        if class_weights is not None:
            self.class_weights = torch.Tensor(class_weights[np.newaxis,:])

        if pool_type=='nopool':
            stride=2
        else:
            stride=1

        self.features = []
        self.features.append(nn.Conv2d(1, 64, 5, stride=stride,padding=2))
        self.features.append(nn.ReLU(True))
        if pool_type=='max':
            self.features.append(nn.MaxPool2d(2,2))
        elif pool_type=='avg':
            self.features.append(nn.AvgPool2d(2,2))
        
        self.features.append(nn.Conv2d(64, 128, 5, stride=stride,padding=2))
        self.features.append(nn.ReLU(True))
        if pool_type=='max':
            self.features.append(nn.MaxPool2d(2,2))
        elif pool_type=='avg':
            self.features.append(nn.AvgPool2d(2,2))
        
        self.features.append(CapsuleLayer(32, 1, 128, 8, kernel_size=7, stride=3, num_iterations=r))
        
        self.features.append(CapsuleLayer(n_classes, 32, 8, 16, kernel_size=6, stride=1, num_iterations=r))
        
        self.features = nn.Sequential(*self.features)

        if self.reconstruct:
            self.reconstruction_loss = nn.MSELoss(size_average=False)
            self.decoder = nn.Sequential(
                nn.Linear(16 * self.num_classes, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 784)
            )
            self.upsampler = nn.Upsample(size=(self.in_size,self.in_size), mode='bilinear')

    def forward(self, data, y = None,return_caps = False):
        
        x = self.features(data).squeeze()
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)

        if self.reconstruct:
            if y is None:
                _, max_length_indices = classes.max(dim=1)
                y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=max_length_indices)
            else:
                y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=y)
        
            
            reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
            reconstructions = reconstructions.view(reconstructions.size(0),1,int(math.sqrt(reconstructions.size(1))),int(math.sqrt(reconstructions.size(1))))
            # print reconstructions.size()
            reconstructions = self.upsampler(reconstructions)
            
            if return_caps:
                return classes, reconstructions, data, x
            else:
                return classes, reconstructions, data
        else:
            if return_caps:
                return classes, x
            else:
                return classes
    
    def margin_loss(self,classes,labels):
        if self.reconstruct:
            images = classes[2]
            reconstructions = classes[1]
            classes = classes[0]

        spread_loss = self.spread_loss(classes,labels,1)

        # spread_loss = spread_loss.sum()

        if self.reconstruct:
            reconstruction_loss = self.reconstruction_loss(reconstructions, images)
            # print reconstruction_loss
            # print spread_loss
            # raw_input()
            total_loss = spread_loss +0.5*0.00005*reconstruction_loss/labels.size(0)
                # +  0.00005*reconstruction_loss)/labels.size(0)
                # ) / images.size(0)
        else:
            total_loss = spread_loss
            # / labels.size(0)

        return total_loss


class Network:
    def __init__(self,n_classes=8, spread_loss_params=None, pool_type='max',r=3, reconstruct=False,init=False,class_weights = None):
        # print 'BN',bn
        if spread_loss_params is None:
            spread_loss_params = {'end_epoch':int(num_epochs*0.9),'decay_steps':5,'init_margin' : 0.9, 'max_margin' : 0.9}

        model = Khorrami_Capsule_Reconstruct(n_classes,spread_loss_params, pool_type,r,reconstruct,class_weights)

        if init:
            for idx_m,m in enumerate(model.features):
                if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                    # print m,1
                    nn.init.xavier_normal(m.weight.data)
                    nn.init.constant(m.bias.data,0.)
                elif isinstance(m, CapsuleLayer):
                    # print m,2
                    if m.num_in_capsules==1:
                        # print m,3
                        nn.init.xavier_normal(m.capsules.weight.data)
                        nn.init.constant(m.capsules.bias.data,0.)
                    else:
                        # print m,4
                        nn.init.normal(m.route_weights.data, mean=0, std=0.1)
                
        self.model = model
        
    
    def get_lr_list(self, lr):
        lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]
        # \
        #         +[{'params': self.model.classifier.parameters(), 'lr': lr[1]}]
        return lr_list


def main():
    import numpy as np
    import torch

    spread_loss_params = dict(end_epoch=int(num_epochs*0.9),decay_steps=5,init_margin = 0.9, max_margin = 0.9)
    net = Network(n_classes= 8, spread_loss_params = spread_loss_params, pool_type='max', reconstruct=True, init = False)
    print net.model
    net.model = net.model.cuda()
    input = np.zeros((10,1,96,96))
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    output,recons,input = net.model(input)

    print output.data.shape
    print recons.data.shape

if __name__=='__main__':
    main()



