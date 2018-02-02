# import sys
# sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
# import models

from matrix_capsules import PrimaryCaps, ConvCaps

# from train_ck import augment_image
import scipy.misc
import numpy as np
# import dataset
from torchvision import transforms

class CapsNet(nn.Module):
    def __init__(self,A=32,B=32,C=32,D=32, E=10,r = 3):
        super(CapsNet, self).__init__()
        self.E = E
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=2)
        self.primary_caps = PrimaryCaps(A,B)
        self.convcaps1 = ConvCaps(B, C, kernel = 3, stride=2,iteration=r,
                                  coordinate_add=False, transform_share = False)
        self.convcaps2 = ConvCaps(C, D, kernel = 3, stride=1,iteration=r,
                                  coordinate_add=False, transform_share = False)
        self.classcaps = ConvCaps(D, E, kernel = 0, stride=1,iteration=r,
                                  coordinate_add=True, transform_share = True) 
        
        
    def forward(self,x): #b,1,28,28
        # print 'input',x.size()
        x = F.relu(self.conv1(x)) #b,32,12,12
        # print 'input',x.size()
        x = self.primary_caps(x) #b,32*(4*4+1),12,12
        # print 'input',x.size()
        x = self.convcaps1(x) #b,32*(4*4+1),5,5
        # print 'input',x.size()
        x = self.convcaps2(x) #b,32*(4*4+1),3,3
        # print 'input',x.size()
        x = self.classcaps(x).view(-1,self.E*16+self.E) #b,10*16+10
        # print 'input',x.size()
        return x
    
    def loss(self, x, target, m): #x:b,10 target:b
        b = x.size(0)
        a_t = torch.cat([x[i][target[i]] for i in range(b)]) #b
        a_t_stack = a_t.view(b,1).expand(b,self.E).contiguous() #b,10
        u = m-(a_t_stack-x) #b,10
        mask = u.ge(0).float() #max(u,0) #b,10
        # print mask

        loss = ((mask*u)**2).sum()/b - m**2  #float

        return loss
    
    def loss2(self,x ,target):
        loss = F.cross_entropy(x,target)
        return loss

    def spread_loss(self,x,target,m):
        use_cuda = next(self.parameters()).is_cuda
        # print 'in spread_loss'
        # print 'x',x
        # print 'target',target
        b = x.size(0)
        target_t = target.type(torch.LongTensor)
        
        if use_cuda:
            target_t = target_t.cuda()
        
        rows = torch.LongTensor(np.array(range(b)))
        
        if use_cuda:
            rows = rows.cuda()

        a_t = x[rows,target_t]
        # print 'a_t',a_t
        a_t_stack = a_t.view(b,1).expand(b,x.size(1)).contiguous() #b,10
        # print 'a_t_stack',a_t_stack
        # print 'x',x
        u = m-(a_t_stack-x) #b,10
        # print 'u',u
        u = nn.functional.relu(u)**2
        u[rows,target_t]=0
        # print 'u',u
        loss = torch.sum(u)/b
        # print 'loss',loss
        # raw_input()
        return loss

class Network:
    def __init__(self,A,B,C,D,E,r):
        # conv_layers = [[32,5,2]]
        # caps_layers = [[8,1,1],[16,3,2],[16,3,2],[16,3,2]]

        model = CapsNet(A=A,B=B,C=C,D=D, E=E,r=r)
        
        # for idx_m,m in enumerate(model.children()):
        #     print m
        #     if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
        #         nn.init.xavier_normal(m.weight.data)
        #         nn.init.constant(m.bias.data,0.)
        #     elif isinstance(m, PrimaryCaps):
        #         print 'hello primary'
        #         nn.init.xavier_normal(m.pose.weight.data)
        #         nn.init.constant(m.pose.bias.data,0.)

        #         nn.init.xavier_normal(m.activation[0].weight.data)
        #         nn.init.constant(m.activation[0].bias.data,0.)
                
        #     elif isinstance(m, ConvCaps):
        #         print 'hello'
                # nn.init.constant(m.beta_v.data,0.)
                # nn.init.constant(m.beta_a.data,0.)
                # nn.init.xavier_normal(m.W.data)
                
        self.model = model
     
    
    def get_lr_list(self, lr):
        # lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]\
        #         +[{'params': self.model.classifier.parameters(), 'lr': lr[1]}]
        lr_list = [{'params':filter(lambda x: x.requires_grad, self.model.parameters()), 'lr': lr}]
        return lr_list

def main():
    import numpy as np
    import torch
    from torch.autograd import Variable
    import torch.optim as optim

    # options = {'mnist': ([[[8., 8.], [12., 8.], [16., 8.]],
    #                       [[8., 12.], [12., 12.], [16., 12.]],
    #                       [[8., 16.], [12., 16.], [16., 16.]]], 28.),
    #            'smallNORB': ([[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
    #                           [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
    #                           [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
    #                           [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]], 32.)
    #            }
    # mnist_coord_add = np.array(options['mnist'][0])
    # print mnist_coord_add.shape

    # smallNORB_coord_add = np.array(options['smallNORB'][0])
    # print smallNORB_coord_add.shape
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(2)

    # print np.array(options['mnist'][0])
    
    
    A,B,C,D,E,r = 32,8,16,16,8,1
    net = Network(A,B,C,D,E,r)
    net.model = net.model.cuda()
    optimizer = optim.SGD(net.get_lr_list(0.02), lr=0, momentum=0.9)

    print net.model
    # raw_input()
    input = np.random.randn(10,1,28,28)
    # np.zeros((10,1,96,96))
    # labels = np.array([0,1,2,3,4,5,6,7,0,1],dtype=np.int)

    labels = np.zeros((10,),dtype=np.int)
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    labels = Variable(torch.Tensor(labels).cuda())
    print labels.size()
    loss_old = 100
    # loss_func = net.model.spread_loss()
    # nn.CrossEntropyLoss() 
    for i in range(100):
        output = net.model(input)
        # print output
        loss = net.model.spread_loss(output,labels,0.2)
        print i,loss.data[0]
        # if loss_old<loss.data[0]:
        #     print output_old
        #     print output
        #     raw_input()

        loss.backward()
        optimizer.step()
        # output_old = output
        # loss_old = loss.data[0]
        # raw_input()
    print output.size()
    print output

if __name__=='__main__':
    main()