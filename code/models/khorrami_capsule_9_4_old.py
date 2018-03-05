from torchvision import models
import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import torch.nn.functional as F
from CapsuleLayer import CapsuleLayer,softmax
from dynamic_capsules import Dynamic_Capsule_Model_Super


class Khorrami_Capsule(Dynamic_Capsule_Model_Super):

    def __init__(self,n_classes,pool_type='max',r=3,class_weights=None):
        super(Dynamic_Capsule_Model_Super, self).__init__()
        
        self.reconstruct = False
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
        
        self.features.append(CapsuleLayer(32, 1, 128, 8, kernel_size=9, stride=4, num_iterations=r))
        
        
        self.features = nn.Sequential(*self.features)
        
        
        self.caps = CapsuleLayer(n_classes, 32, 8, 16, kernel_size=4, stride=1, num_iterations=r)
        
    # def forward(self, x):
    #     x = self.features(x)
    #     print x.size()
    #     return x

    def forward(self, data, y = None,return_caps = False):
        x = self.features(data)
        x = self.caps(x)
        x = x.squeeze()
        
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)
        if return_caps:
            return classes, x
        else:
            return classes

class Network:
    def __init__(self,n_classes=8,pool_type='max',r=3, init=False,class_weights = None):
        # print 'BN',bn
        model = Khorrami_Capsule(n_classes,pool_type,r,class_weights)

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
        return lr_list


def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    net = Network(n_classes= 8, pool_type='nopool', init = True)
    print net.model
    net.model = net.model.cuda()
    input = np.zeros((10,1,96,96))
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    output = net.model(input)
    print output.data.shape

if __name__=='__main__':
    main()



