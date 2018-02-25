from torchvision import models
import torch.nn as nn

import numpy as np
import scipy.misc
import torch
from CapsuleLayer import CapsuleLayer
from dynamic_capsules import Dynamic_Capsule_Model_Super
from spread_loss import Spread_Loss
import torch.nn.functional as F

class Vgg_Capsule_Disfa(Dynamic_Capsule_Model_Super):

    def __init__(self,n_classes,r=3,class_weights=None):
        super(Dynamic_Capsule_Model_Super, self).__init__()
        
        self.vgg_base = torch.load('models/pytorch_vgg_face_just_conv.pth')
        # print self.vgg_base

        self.reconstruct = False
        if class_weights is not None:
            self.class_weights = torch.Tensor(class_weights[np.newaxis,:])

        self.features = []
        
        self.features.append(CapsuleLayer(32, 1, 512, 8, kernel_size=3, stride=2, num_iterations=r))
        
        self.features.append(CapsuleLayer(n_classes, 32, 8, 16, kernel_size=6, stride=1, num_iterations=r))
        
        self.features = nn.Sequential(*self.features)
        
    def forward(self,data, y = None,return_caps = False):
        x = self.vgg_base(data)
        # print torch.min(x),torch.max(x)
        # raw_input()
        # print x.shape
        x = self.features(x)
        # .squeeze()
        # print x.shape
        x = x.squeeze()
        # print x.shape

        classes = (x ** 2).sum(dim=-1) ** 0.5
        # print torch.min(classes),torch.max(classes)
        # classes = F.tanh(classes)

        if self.reconstruct:
            if y is None:
                _, max_length_indices = classes.max(dim=1)
                y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=max_length_indices)
            else:
                y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=y)
            
            reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
            reconstructions = reconstructions.view(reconstructions.size(0),1,int(math.sqrt(reconstructions.size(1))),int(math.sqrt(reconstructions.size(1))))
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
    def __init__(self,n_classes=8,r=3, init=False,class_weights = None):
        # print 'BN',bn
        model = Vgg_Capsule_Disfa(n_classes,r,class_weights)
        
        if init:
            for idx_m,m in enumerate(model.features):
                if isinstance(m, CapsuleLayer):
                    
                    if m.num_in_capsules==1:
                        nn.init.xavier_normal(m.capsules.weight.data)
                        nn.init.constant(m.capsules.bias.data,0.)
                    else:
                        nn.init.normal(m.route_weights.data, mean=0, std=0.1)
                elif isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                    # print m,1

                    # print m.weight.data.shape, torch.min(m.weight.data), torch.max(m.weight.data)
                    # print m.bias.data.shape, torch.min(m.bias.data), torch.max(m.bias.data)

                    nn.init.xavier_normal(m.weight.data)
                    nn.init.constant(m.bias.data,0.)

                
        self.model = model
        
    
    def get_lr_list(self, lr):
        lr_list= [{'params': self.model.vgg_base.parameters(), 'lr': lr[0]}] +\
                [{'params': self.model.features.parameters(), 'lr': lr[1]}]
        return lr_list


def main():
    import numpy as np
    import torch
    from torch.autograd import Variable
    
    n_classes = 10
    net = Network(n_classes= n_classes, init = True)
    print net.model
    labels = np.ones((10,n_classes))
    net.model = net.model.cuda()
    input = np.zeros((10,3,224,224))
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    labels = Variable(torch.LongTensor(labels).cuda())
    output = net.model(input)
    print output.data.shape
    criterion = nn.MultiLabelSoftMarginLoss()
    criterion(output,labels)

    # criterion = Spread_Loss(50,5)
    # for epoch_num in range(53):
    #     print epoch_num,criterion(output,labels,epoch_num)



if __name__=='__main__':
    main()



