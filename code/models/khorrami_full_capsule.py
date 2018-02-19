from torchvision import models
import torch.nn as nn

import numpy as np
import scipy.misc
import torch
from CapsuleLayer import CapsuleLayer
from dynamic_capsules import Dynamic_Capsule_Model_Super


class Khorrami_Full_Capsule(Dynamic_Capsule_Model_Super):

    def __init__(self,n_classes,conv_layers, caps_layers,r=3):
        super(Dynamic_Capsule_Model_Super, self).__init__()
        
        self.reconstruct = False
        self.features = []
        for conv_param in conv_layers:
            self.features.append(nn.Conv2d(in_channels=1, out_channels=conv_param[0],
                                   kernel_size=conv_param[1], stride=conv_param[2]))
            self.features.append(nn.ReLU(True))

        for idx_caps_param,caps_param in enumerate(caps_layers):

          num_capsules, out_channels,kernel_size, stride = caps_param

          if idx_caps_param==0:
              in_channels = conv_layers[-1][0]
              num_in_capsules = 1
          else:
              num_in_capsules = caps_layers[idx_caps_param-1][0]
              in_channels = caps_layers[idx_caps_param-1][1]

          print num_capsules, num_in_capsules, in_channels, out_channels, kernel_size, stride, r

          self.features.append(CapsuleLayer(num_capsules, num_in_capsules, in_channels, out_channels, kernel_size=kernel_size, stride=stride, num_iterations=r))

        
        self.features = nn.Sequential(*self.features)

        # if pool_type=='nopool':
        #     stride=2
        # else:
        #     stride=1

        # self.features = []
        # self.features.append(nn.Conv2d(1, 32, 5, stride=stride,padding=2))
        # self.features.append(nn.ReLU(True))
        # if pool_type=='max':
        #     self.features.append(nn.MaxPool2d(2,2))
        # elif pool_type=='avg':
        #     self.features.append(nn.AvgPool2d(2,2))
        
        # self.features.append(nn.Conv2d(32, 64, 5, stride=stride,padding=2))
        # self.features.append(nn.ReLU(True))
        # if pool_type=='max':
        #     self.features.append(nn.MaxPool2d(2,2))
        # elif pool_type=='avg':
        #     self.features.append(nn.AvgPool2d(2,2))
        
        # self.features.append(CapsuleLayer(32, 1, 64, 8, kernel_size=7, stride=3, num_iterations=r))
        
        # self.features.append(CapsuleLayer(n_classes, 32, 8, 16, kernel_size=6, stride=1, num_iterations=r))
        
        # self.features = nn.Sequential(*self.features)
        
    # def forward(self, x):
    #     x = self.features(x)
    #     print x.size()
    #     return x

class Network:
    def __init__(self,n_classes=8,conv_layers = None,caps_layers=None,r=3, init=False):
        # print 'BN',bn
        if conv_layers is None:
            conv_layers = [[64,5,2]]
        if caps_layers is None:
            caps_layers = [[16,8,5,2],[32,8,7,3],[n_classes,16,5,1]]

        model = Khorrami_Full_Capsule(n_classes,conv_layers, caps_layers,r)

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
    from torch.autograd import Variable

    net = Network(n_classes= 8)
    print net.model
    net.model = net.model.cuda()
    input = np.zeros((4,1,96,96))
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    output = net.model(input)
    print output.data.shape

if __name__=='__main__':
    main()



