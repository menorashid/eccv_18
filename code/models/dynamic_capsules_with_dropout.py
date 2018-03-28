import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import math
from torch.autograd import Variable

import torch.nn.functional as F
from CapsuleLayer import CapsuleLayer, softmax
from dynamic_capsules import Dynamic_Capsule_Model_Super

class Dynamic_Capsule_Model(Dynamic_Capsule_Model_Super):

    def __init__(self,
                n_classes,
                r,
                reconstruct = False,
                class_weights=None,
                loss_weights = None,
                dropout = 0):
        super(Dynamic_Capsule_Model, self).__init__()
        self.class_weights = class_weights
        self.num_classes = n_classes
        self.reconstruct = reconstruct
        self.loss_weights = loss_weights
        

        # if conv_layers is None:
        #     conv_layers = [[256,9,1]]
        # if caps_layers is None:
        #     caps_layers = [[32,8,9,2],[n_classes,16,6,1]]


        self.features = []
        self.features.append(nn.Conv2d(in_channels=1, out_channels=256,
                               kernel_size=9, stride=1))
        self.features.append(nn.ReLU(True))
        self.features.append(CapsuleLayer(32, 1, 256, 8, kernel_size=9, stride=2, num_iterations=r, dropout = 0))
        self.features = nn.Sequential(*self.features)

        self.caps = CapsuleLayer(n_classes, 32, 8, 16, kernel_size=6, stride=1, num_iterations=r, dropout = dropout)

        

        if self.reconstruct:
            self.reconstruction_loss = nn.MSELoss(size_average=False)
            self.decoder = nn.Sequential(
                nn.Linear(out_channels * self.num_classes, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 784),
            )

    def forward(self, data, y = None,return_caps = False):
        # print 'IN FORWARD',self.reconstruct,data.size(),y.size(),
        
        x = self.features(data)
        x = self.caps(x).squeeze()
        
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
    def __init__(self,n_classes=10,r=3,conv_layers = None, caps_layers = None,reconstruct=False, init= False, loss_weights = None, dropout = 0):
        
        model = Dynamic_Capsule_Model(n_classes,r,reconstruct=reconstruct, loss_weights = loss_weights, dropout = dropout)

        if init:
            for idx_m,m in enumerate(model.modules()):
                if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                    # print m
                    nn.init.xavier_normal(m.weight.data)
                    nn.init.constant(m.bias.data,0.)
                elif isinstance(m, CapsuleLayer):
                    if m.num_in_capsules==1:
                        nn.init.normal(m.capsules.weight.data,std=0.1)
                        nn.init.constant(m.capsules.bias.data,0.)
                    else:
                        nn.init.normal(m.route_weights.data, mean=0, std=0.1)
                        nn.init.constant(m.bias.data,0.)
                        
                
        self.model = model

    def get_lr_list(self, lr):
        
        lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]
        lr_list = lr_list + [{'params': self.model.caps.parameters(), 'lr': lr[1]}]
        if self.model.reconstruct:
            lr_list= lr_list + [{'params': self.model.decoder.parameters(), 'lr': lr[2]}]
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
    
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    lr = [0.001]
    decay_rate = 0.96
    decay_steps = 469
    min_lr = 1e-6
    optimizer = Adam(network.get_lr_list(lr))
    # exp_lr_scheduler = Exp_Lr_Scheduler(optimizer,0,lr,decay_rate,decay_steps,min_lr)

    batch_size = 4
    test_batch_size = 4

    kwargs = {'num_workers': 1, 'pin_memory': True}
     # if args.cuda else {}

    transformer = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    train_data = datasets.MNIST('../../data/mnist_downloaded', train=True, transform = transformer)
    test_data = datasets.MNIST('../../data/mnist_downloaded', train=False, download = True, transform = transformer)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=test_batch_size, shuffle=True, **kwargs)

    class_weights = np.array([0.1]*10)
    print class_weights
    model.class_weights = class_weights
    disp_after = 1
    num_epochs = 10
    model.train()
    for epoch_num in range(num_epochs):
        for batch_idx, (data, labels) in enumerate(train_loader):
            
            

            # print data.shape, torch.min(data), torch.max(data)
            # print labels.shape, torch.min(labels), torch.max(labels)
            
            # print labels.shape, torch.min(labels), torch.max(labels)
            
            # labels_simple = torch.sparse.torch.eye(num_classes).index_select(dim=0, index=labels)
            labels_simple = labels
            data, labels = data.cuda(), labels.cuda()
            data, labels = Variable(data), Variable(labels)
            optimizer.zero_grad()
            
            classes = model(data)
            # classes, reconstructions = model(data, labels)
            # raw_input()
            # print classes.shape, reconstructions.shape
            # else:
            #     classes, reconstructions = model(data)

            loss = model.spread_loss(classes, labels)
            # margin_loss( classes,labels)
            # 
            # 
            # # , reconstructions)

            if batch_idx % disp_after ==0:  
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_num, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

            
            loss.backward()
            optimizer.step()
            step_curr = len(train_loader)*epoch_num+batch_idx
            exp_lr_scheduler.step()
            # (optimizer, step_curr, lr, decay_rate, decay_steps, min_lr = min_lr)
            print step_curr,optimizer.param_groups[-1]['lr']
            
        
             




    

if __name__=='__main__':
    main()