from torchvision import models
import torch.nn as nn

import numpy as np
import scipy.misc
import torch
from CapsuleLayer import CapsuleLayer
from dynamic_capsules import Dynamic_Capsule_Model_Super
from spread_loss import Spread_Loss
import torch.nn.functional as F

class Dummy_Model(nn.Module):

    def __init__(self,n_classes,r=3,class_weights=None):
        super(Dummy_Model, self).__init__()
        
        stride=1
        
        self.vgg_base = []
        self.vgg_base.append(nn.Conv2d(3, 64, 5, stride=stride,padding=2))
        self.vgg_base.append(nn.ReLU(True))
        self.vgg_base.append(nn.MaxPool2d(2,2))
        
        self.vgg_base.append(nn.Conv2d(64, 128, 5, stride=stride,padding=2))
        self.vgg_base.append(nn.ReLU(True))
        self.vgg_base.append(nn.MaxPool2d(2,2))

        self.vgg_base.append(nn.Conv2d(128, 256, 7, stride=3))
        self.vgg_base.append(nn.ReLU(True))
        self.vgg_base = nn.Sequential(*self.vgg_base)

        self.features = []
        self.features.append(nn.Linear(256*6*6,n_classes))
        self.features = nn.Sequential(*self.features)
        
    def forward(self,data, y = None,return_caps = False):
        x = self.vgg_base(data)
        x = x.view(x.size(0), 256*6*6)
        x = self.features(x)
        return x



class Network:
    def __init__(self,n_classes=8,r=3, init=True,class_weights = None):
        # print 'BN',bn
        model = Dummy_Model(n_classes,r,class_weights)
        
        if init:
            for idx_m,m in enumerate(model.features):
                if isinstance(m, CapsuleLayer):
                    
                    if m.num_in_capsules==1:
                        nn.init.xavier_normal(m.capsules.weight.data)
                        nn.init.constant(m.capsules.bias.data,0.)
                    else:
                        nn.init.normal(m.route_weights.data, mean=0, std=0.1)
                elif isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
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
    import torch.optim as optim

    n_classes = 10
    net = Network(n_classes= n_classes, init = True)
    print net.model
    labels = np.random.randn(16,n_classes)
    labels[labels>0.5]=1
    labels[labels<0.5]=0

    net.model = net.model.cuda()
    input = np.random.randn(16,3,96,96)
    
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    optimizer = optim.Adam(net.model.parameters())
    labels = Variable(torch.FloatTensor(labels).cuda())
    # output = net.model(input)
    # print output.data.shape
    criterion = nn.MultiLabelSoftMarginLoss()
    # criterion(output,labels)
    epochs = 50
    for epoch in range(epochs):
        # inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
        # labelsv = Variable(torch.FloatTensor(labels[i])).view(1, -1)
        # print labelsv
        output = net.model(input)
        loss = criterion(output, labels)
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



