from torchvision import models
import torch.nn as nn

import numpy as np
import scipy.misc
import torch
from CapsuleLayer import CapsuleLayer
from dynamic_capsules import Dynamic_Capsule_Model_Super


class Caps_For_96(Dynamic_Capsule_Model_Super):

    def __init__(self,n_classes,conv_layers, caps_layers,r=3,pool_type='max'):
        super(Dynamic_Capsule_Model_Super, self).__init__()
        
        self.reconstruct = False
        self.features = []
        for idx_conv_param,conv_param in enumerate(conv_layers):
            if idx_conv_param==0:
                in_channels = 1
            else:
                in_channels = conv_layers[idx_conv_param-1][0]
            self.features.append(nn.Conv2d(in_channels=in_channels, out_channels=conv_param[0],
                                   kernel_size=conv_param[1], stride=conv_param[2],padding=conv_param[3]))
            self.features.append(nn.ReLU(True))
            if pool_type=='max':
                self.features.append(nn.MaxPool2d(2,2))
            elif pool_type=='avg':
                self.features.append(nn.AvgPool2d(2,2))

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

    # def forward(self, x):
    #     x = self.features(x)
    #     print x.size()
    #     return x

class Network:
    def __init__(self,n_classes=8,conv_layers = None,caps_layers=None,r=3, init=False,pool_type='max'):
        # print 'BN',bn
        if conv_layers is None:
            conv_layers = [[256,11,5,5]]
        if caps_layers is None:
            # caps_layers =[]
            caps_layers = [[32,8,9,2],[n_classes,16,6,1]]

        model = Caps_For_96(n_classes,conv_layers, caps_layers,r,pool_type)

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
                        nn.init.constant(m.bias.data,0.)
                
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
    n_classes= 8
    conv_layers = [[256,7,3,0]]
    # ,[128,5,2,0]]
    # ,[64,5,2,0],[128,5,2,0]]
    caps_layers = [[32,8,7,3],[n_classes,32,3,1]]
    net = Network(n_classes= n_classes,conv_layers=conv_layers,caps_layers=caps_layers,pool_type='nopool')
    print net.model
    net.model = net.model.cuda()
    input = np.zeros((4,1,48,48))
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    output = net.model(input)
    print output.data.shape

if __name__=='__main__':
    main()



