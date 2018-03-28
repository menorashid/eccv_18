from torchvision import models
import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import torch.nn.functional as F
from CapsuleLayer import CapsuleLayer,softmax
from dynamic_capsules import Dynamic_Capsule_Model_Super
from torch.autograd import Variable
import math


class Khorrami_Capsule(Dynamic_Capsule_Model_Super):

    def __init__(self,n_classes,pool_type='max',r=3,class_weights=None, reconstruct = False,loss_weights=None, dropout = 0):
        super(Dynamic_Capsule_Model_Super, self).__init__()
        
        self.reconstruct = reconstruct 
        self.num_classes = n_classes
        self.loss_weights = loss_weights
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
        
        self.features.append(CapsuleLayer(32, 1, 128, 8, kernel_size=7, stride=3, num_iterations=r, dropout = 0))
        
        
        self.features = nn.Sequential(*self.features)
        
        
        self.caps = CapsuleLayer(n_classes, 32, 8, 32, kernel_size=6, stride=1, num_iterations=r, dropout = dropout)
            
        if self.reconstruct:
            self.reconstruction_loss = nn.MSELoss(size_average=False)
            self.decoder = nn.Sequential(
                nn.Linear(32 * self.num_classes, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
            )
            self.upsampler = nn.Upsample(size=(96,96), mode='bilinear')

    
    def forward(self, data, y = None,return_caps = False):
        x = self.features(data)
        x = self.caps(x)
        x = x.squeeze()
        
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
            reconstructions = self.upsampler(reconstructions)

            # print reconstructions.size(),torch.min(reconstructions),torch.max(reconstructions)
            # print data.size(),torch.min(data),torch.max(data)
            # raw_input()
            if return_caps:
                return classes, reconstructions, data, x
            else:
                return classes, reconstructions, data
        else:
            if return_caps:
                return classes, x
            else:
                return classes

class Network:
    def __init__(self,n_classes=8,pool_type='max',r=3, init=False,class_weights = None,reconstruct = False,loss_weights = None, dropout = 0):
        # print 'BN',bn
        model = Khorrami_Capsule(n_classes,pool_type,r,class_weights,reconstruct,loss_weights, dropout = dropout)

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
                        nn.init.normal(m.route_weights.data, mean=0, std=1)
                
        self.model = model
        
    
    def get_lr_list(self, lr):
        lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]+[{'params': self.model.caps.parameters(), 'lr': lr[1]}]
        if self.model.reconstruct:
            lr_list = lr_list + [{'params': self.model.decoder.parameters(), 'lr': lr[2]}]
            
        return lr_list


def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    net = Network(n_classes= 8, pool_type='nopool', init = False, reconstruct = True)
    print net.model
    net.model = net.model.cuda()
    input = np.zeros((10,1,96,96))
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    output,recon,data = net.model(input)
    print output.data.shape,recon.data.shape,data.data.shape

if __name__=='__main__':
    main()



