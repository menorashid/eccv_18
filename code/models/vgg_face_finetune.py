from torchvision import models
import torch.nn as nn

import numpy as np
import scipy.misc
import torch
from torch.autograd import Variable
import math


class Vgg_Face(nn.Module):

    def __init__(self,n_classes):
        super(Vgg_Face, self).__init__()
        

        # vgg16 = models.vgg16()
        # vgg_base = list(vgg16.features.children())
        # fc = list(vgg16.classifier.children())
        
        import VGG_FACE

        vgg16 = VGG_FACE.VGG_FACE
        vgg16.load_state_dict(torch.load('models/VGG_FACE.pth'))

        # print vgg16

        vgg_layers = list(vgg16.children())
        self.vgg_base = nn.Sequential(*vgg_layers[:24])
        self.vgg_last_conv = nn.Sequential(*vgg_layers[24:31])

        self.fc = nn.Sequential(*[list(vgg_layers[32])[1], 
                vgg_layers[33],
                vgg_layers[34], 
                list(vgg_layers[35])[1],
                vgg_layers[36], 
                vgg_layers[37]])
        self.last_fc = nn.Linear(4096,n_classes)

        # print fcs
        # # print self.vgg_base
        # # print self.vgg_last_conv
        # print vgg_layers[31:]

        # raw_input()

        # self.vgg_base = nn.Sequential(*vgg_base[:24])
        # self.vgg_last_conv = nn.Sequential(*vgg_base[24:])
        # self.fc = nn.Sequential(*fc[:6])
        # self.last_fc = nn.Linear(4096,n_classes)

    
    def forward(self, data, y = None,return_caps = False):
        x = self.vgg_base(data)
        x = self.vgg_last_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.last_fc(x)
        
        return x

    


class Network:
    def __init__(self,n_classes=8,to_init=[]):
        # print 'BN',bn
        model = Vgg_Face(n_classes)

        if 'vgg_base' in to_init:
            for idx_m,m in enumerate(model.vgg_base):
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal(m.weight.data)
                    nn.init.constant(m.bias.data,0.)
            

        if 'vgg_last_conv' in to_init:
            for idx_m,m in enumerate(model.vgg_last_conv):
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal(m.weight.data)
                    nn.init.constant(m.bias.data,0.)
        
        if 'fc' in to_init:
            for idx_m,m in enumerate(model.fc):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal(m.weight.data)
                    nn.init.constant(m.bias.data,0.)
        
        if 'last_fc' in to_init:
            m = model.last_fc
            # for idx_m,m in enumerate(model.last_fc):
            #     if isinstance(m, nn.Linear):
            nn.init.xavier_normal(m.weight.data)
            nn.init.constant(m.bias.data,0.)

        self.model = model
        
    
    def get_lr_list(self, lr):

        lr_list =[]
        module_list = [self.model.vgg_base,self.model.vgg_last_conv,self.model.fc,self.model.last_fc]
        
        for lr_curr,param_set in zip(lr,module_list):
            if lr_curr==0:
                for param in param_set.parameters():
                    param.requires_grad = False
            else:
                lr_list.append({'params': param_set.parameters(), 'lr': lr_curr})
        
        return lr_list


def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    net = Network(n_classes= 8)
    # , pool_type='nopool', init = False, reconstruct = True)
    print net.model
    net.model = net.model.cuda()
    input = np.zeros((10,3,224,224))
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    output = net.model(input)
    print output.data.shape
    # ,recon.data.shape,data.data.shape

if __name__=='__main__':
    main()



