from torchvision import models
import torch.nn as nn

import numpy as np
import scipy.misc
import torch
    

class Khorrami_Model(nn.Module):

    def __init__(self,n_classes,bn):
        super(Khorrami_Model, self).__init__()
        self.features = []
        self.features.append(nn.Conv2d(1, 64, 5, padding = 2))
        self.features.append(nn.ReLU(True))
        if bn:
            self.features.append(nn.BatchNorm2d(64,affine=True,momentum=0.1))
        self.features.append(nn.MaxPool2d(2,2))
        
        self.features.append(nn.Conv2d(64, 128, 5, padding = 2))
        self.features.append(nn.ReLU(True))
        if bn:
            self.features.append(nn.BatchNorm2d(128,affine=True,momentum=0.1))
        self.features.append(nn.MaxPool2d(2,2))
        
        self.features.append(nn.Conv2d(128, 256, 5, padding = 2))
        self.features.append(nn.ReLU(True))
        if bn:
            self.features.append(nn.BatchNorm2d(256,affine=True,momentum=0.1))
        self.features.append(nn.AvgPool2d(12,12)) # quadrant pooling
        
        self.features = nn.Sequential(*self.features)
        self.classifier = []
        self.classifier.append(nn.Linear(1024,300))
        self.classifier.append(nn.ReLU(True))
        self.classifier.append(nn.Dropout(0.5))
        if bn:
            self.classifier.append(nn.BatchNorm1d(300,affine=True,momentum=0.1))
        self.classifier.append(nn.Linear(300,n_classes))
        
        self.classifier = nn.Sequential(*self.classifier)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 1024)
        x = self.classifier(x)
        return x

class Network:
    def __init__(self,n_classes=8,bn=False, init=True):
        print 'BN',bn
        model = Khorrami_Model(n_classes,bn)

        if init:
            for idx_m,m in enumerate(model.modules()):
                if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                    # print m,1

                    # print m.weight.data.shape, torch.min(m.weight.data), torch.max(m.weight.data)
                    # print m.bias.data.shape, torch.min(m.bias.data), torch.max(m.bias.data)

                    nn.init.xavier_normal(m.weight.data)
                    nn.init.constant(m.bias.data,0.)

                    # print m.weight.data.shape, torch.min(m.weight.data), torch.max(m.weight.data)
                    # print m.bias.data.shape, torch.min(m.bias.data), torch.max(m.bias.data)

                elif isinstance(m, nn.BatchNorm2d):
                    # print m,2
                    nn.init.constant(m.weight.data,1.)
                    nn.init.constant(m.bias.data,0.)
                # print 'break'
                
        self.model = model
        
    
    def get_lr_list(self, lr):
        lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]\
                +[{'params': self.model.classifier.parameters(), 'lr': lr[1]}]
        return lr_list


def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    net = Network(8)
    print net.model
    input = np.zeros((10,1,96,96))
    input = torch.Tensor(input)
    print input.shape
    input = Variable(input)
    output = net.model(input)
    print output.data.shape

if __name__=='__main__':
    main()



