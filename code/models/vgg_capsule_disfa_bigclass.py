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

    def __init__(self,n_classes,r=3):
        super(Dynamic_Capsule_Model_Super, self).__init__()
        
        self.vgg_base = torch.load('models/pytorch_vgg_face_just_conv.pth')
        # print self.vgg_base

        self.reconstruct = False
        
        self.features = []
        
        self.features.append(CapsuleLayer(32, 1, 512, 8, kernel_size=3, stride=2, num_iterations=r))
        
        self.features.append(CapsuleLayer(n_classes, 32, 8, 32, kernel_size=6, stride=1, num_iterations=r))
        
        self.features = nn.Sequential(*self.features)
        
    def forward(self,data, y = None,return_caps = False):
        x = self.vgg_base(data)
        # print torch.min(x),torch.max(x)

        x = self.features(x).squeeze()
        classes = (x ** 2).sum(dim=-1) ** 0.5
        # classes = F.softmax(classes)
        # print classes.size()
        # raw_input()
        # classes = classes.squeeze()
        # print classes.size()
        # classes = F.softmax(classes)

        return classes



class Network:
    def __init__(self,n_classes=8,r=3, init=False):
        # print 'BN',bn
        model = Vgg_Capsule_Disfa(n_classes,r)
        
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
        for lr_curr,param_set in zip(lr,[self.model.vgg_base,self.model.features]):
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
    net = Network(n_classes= n_classes, init = False)
    print net.model
    labels = np.random.randn(16,n_classes)
    labels[labels>0.5]=1
    labels[labels<0.5]=0

    net.model = net.model.cuda()
    print net.model
    input = np.random.randn(16,3,224,224)
    
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    optimizer = optim.Adam(net.model.parameters(),lr=0.00005)
    labels = Variable(torch.FloatTensor(labels).cuda())
    # output = net.model(input)
    # print output.data.shape
    criterion = nn.MultiLabelSoftMarginLoss()
    # criterion(output,labels)
    epochs = 1000
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



