import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import math
from torch.autograd import Variable

import torch.nn.functional as F

class Convolution_Baseline(nn.Module):

    def __init__(self,n_classes):
        super(Convolution_Baseline, self).__init__()
        self.features = []
        self.features.append(nn.Conv2d(1, 512, 5))
        self.features.append(nn.ReLU(True))
        self.features.append(nn.MaxPool2d(2,2))
        
        self.features.append(nn.Conv2d(512, 256, 5))
        self.features.append(nn.ReLU(True))
        self.features.append(nn.MaxPool2d(2,2))
        
        self.features = nn.Sequential(*self.features)
        self.classifier = []
        self.classifier.append(nn.Linear(256*4*4,1024))
        self.classifier.append(nn.ReLU(True))
        self.classifier.append(nn.Linear(1024,n_classes))
        self.classifier = nn.Sequential(*self.classifier)

        
    def forward(self, x, y = None,return_caps = False):
        x = self.features(x)
        # print x.size()
        x = x.view(x.size(0), 256*4*4)
        x = self.classifier(x)
        return x



class Network:
    def __init__(self,n_classes=10):
        model = Convolution_Baseline(n_classes)

        
        for idx_m,m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                nn.init.constant(m.bias.data,0.)
                
        self.model = model

    def get_lr_list(self, lr):
        lr_list = [{'params':filter(lambda x: x.requires_grad, self.model.parameters()), 'lr': lr}]
        return lr_list


def main():
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchvision import datasets, transforms


    reconstruct =False
    num_classes = 10
    network = Network(num_classes)
    model = network.model
    print model
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))
    optimizer = Adam(network.get_lr_list(0.001))
    
    batch_size = 128
    test_batch_size = 128

    kwargs = {'num_workers': 1, 'pin_memory': True}
    transformer = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    train_data = datasets.MNIST('../../data/mnist_downloaded', train=True, transform = transformer)
    test_data = datasets.MNIST('../../data/mnist_downloaded', train=False, download = True, transform = transformer)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=test_batch_size, shuffle=False, **kwargs)

    disp_after = 1
    num_epochs = 10
    model.train()

    for epoch_num in range(num_epochs):
        for batch_idx, (data, labels) in enumerate(train_loader):
            
            

            # print data.shape, torch.min(data), torch.max(data)
            # print labels.shape, torch.min(labels), torch.max(labels)
            
            # print labels.shape, torch.min(labels), torch.max(labels)
            
            # labels = torch.sparse.torch.eye(num_classes).index_select(dim=0, index=labels)
            data, labels = data.cuda(), labels.cuda()
            data, labels = Variable(data), Variable(labels)
            optimizer.zero_grad()
            

            classes = model(data, labels)
            # raw_input()
            # print classes.shape, reconstructions.shape
            # else:
            #     classes, reconstructions = model(data)

            loss = criterion( classes,labels)
            # # , reconstructions)

            if batch_idx % disp_after ==0:  
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_num, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

            
            loss.backward()
            optimizer.step()
            step_curr = len(train_loader)*epoch_num+batch_idx
            # exp_lr_scheduler.step()
            # (optimizer, step_curr, lr, decay_rate, decay_steps, min_lr = min_lr)
            # print step_curr,optimizer.param_groups[-1]['lr']
            
        
             




    

if __name__=='__main__':
    main()