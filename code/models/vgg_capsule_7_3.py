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


class Vgg_Capsule(Dynamic_Capsule_Model_Super):

    def __init__(self,n_classes,pool_type='max',r=3,class_weights=None, reconstruct = False, loss_weights = None, std_div = None):
        super(Dynamic_Capsule_Model_Super, self).__init__()
        
        self.reconstruct = reconstruct 
        self.num_classes = n_classes
        self.loss_weights = loss_weights
        # self.temp_loss = nn.MultiLabelMarginLoss()

        if class_weights is not None:
            self.class_weights = torch.Tensor(class_weights[np.newaxis,:])

        if std_div is not None:
            self.std_div = torch.Tensor(std_div)

        if pool_type=='nopool':
            stride=2
        else:
            stride=1


        self.vgg_base = torch.load('models/pytorch_vgg_face_just_conv.pth')
        # self.vgg_base = []
        # self.vgg_base.append(nn.Conv2d(3, 64, 5, stride=stride,padding=2))
        # self.vgg_base.append(nn.ReLU(True))
        # self.vgg_base.append(nn.MaxPool2d(2,2))
        # self.vgg_base.append(nn.Conv2d(64, 128, 5, stride=stride,padding=2))
        # self.vgg_base.append(nn.ReLU(True))
        # self.vgg_base.append(nn.MaxPool2d(2,2))
        # self.vgg_base = nn.Sequential(*self.vgg_base)


        self.features = []
        self.features.append(CapsuleLayer(32, 1, 512, 8, kernel_size=3, stride=2, num_iterations=r))
        self.features.append(CapsuleLayer(n_classes, 32, 8, 32, kernel_size=6, stride=1, num_iterations=r))
        self.features = nn.Sequential(*self.features)
        
            
        if self.reconstruct:
            self.reconstruction_loss = nn.MSELoss(size_average=False)
            self.decoder = nn.Sequential(
                nn.Linear(32 * self.num_classes, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 56*56*3),
            )
            self.upsampler = nn.Upsample(size=(224,224), mode='bilinear')

    
    def forward(self, data, y = None,return_caps = False):
        x = self.vgg_base(data)
        # print x.size()
        x = self.features(x)
        # print x.size()
        x = x.squeeze()
        
        classes = (x ** 2).sum(dim=-1) ** 0.5
        # classes = F.sigmoid(classes)
        if self.reconstruct:
            if y is None:
                y = F.relu(classes - 0.5)
                y = torch.ceil(y) 

                # y 
                # # _, max_length_indices = classes.max(dim=1)
                # y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=max_length_indices)
            
            # y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=y)
            # print y.size()
            
            reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
            reconstructions = reconstructions.view(reconstructions.size(0),3,56,56)
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

    def margin_loss(self,  classes,labels):
        if self.reconstruct:
            images = classes[2]
            reconstructions = classes[1]
            classes = classes[0]

        is_cuda = next(self.parameters()).is_cuda
        batch_size = labels.size(0)

        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        # print classes.size(),type(classes.data),type(labels.data)
        # margin_loss = self.temp_loss(classes,labels)

        if hasattr(self, 'class_weights') and self.class_weights is not None:
            # print margin_loss.size(),type(margin_loss)
            
            # print class_weights.size()
            if is_cuda:
                class_weights = torch.autograd.Variable(self.class_weights.cuda())
            else:
                class_weights = torch.autograd.Variable(self.class_weights)

            # print margin_loss.size(),class_weights.size()
            margin_loss = margin_loss*class_weights

        # print margin_loss[0]
        # raw_input()
        margin_loss = margin_loss.sum()
        margin_loss = margin_loss/ batch_size
        
        if self.reconstruct:
            images_copy = Variable(images.data)
            if hasattr(self, 'std_div') and self.std_div is not None:
                for dim_curr in range(3):
                    images_copy[:,dim_curr,:,:]=torch.div(images_copy[:,dim_curr,:,:],self.std_div[dim_curr])
                    

            reconstruction_loss = self.reconstruction_loss(reconstructions, images_copy)
            reconstruction_loss = (0.00001 * reconstruction_loss)/batch_size
            # reconstruction_loss = reconstruction_loss/batch_size
            # (0.0000001 * reconstruction_loss)/batch_size

            if self.loss_weights is not None:
                reconstruction_loss = self.loss_weights[1]*reconstruction_loss
                margin_loss = self.loss_weights[0]*margin_loss
            
            total_loss = margin_loss + reconstruction_loss
            return total_loss, margin_loss, reconstruction_loss
        else:
            total_loss = margin_loss
            return total_loss, margin_loss, margin_loss


class Network:
    def __init__(self,n_classes=8,pool_type='max',r=3, init=False,class_weights = None,reconstruct = False,loss_weights = None, std_div = None):
        # print 'BN',bn
        model = Vgg_Capsule(n_classes,pool_type,r,class_weights,reconstruct,loss_weights, std_div = std_div)

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

        lr_list =[]
        module_list = [self.model.vgg_base,self.model.features]
        if self.model.reconstruct:
            module_list.append(self.model.decoder)

        # print lr_list
        # print len(module_list) 
        for lr_curr,param_set in zip(lr,module_list):
            # print lr
            if lr_curr==0:
                for param in param_set.parameters():
                    param.requires_grad = False
                # lr_list.append({'params':None, 'lr': lr_curr})
            else:
                lr_list.append({'params': param_set.parameters(), 'lr': lr_curr})
        # print lr_list
        # raw_input()
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



