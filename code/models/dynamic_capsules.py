import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import math
from torch.autograd import Variable

import torch.nn.functional as F

# from dynamic_capsule_layer import CapsuleLayer

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_in_capsules, in_channels, out_channels, kernel_size, stride,
                 num_iterations=3):
        
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_in_capsules*kernel_size*kernel_size
        self.num_in_capsules = num_in_capsules
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels

        if self.num_in_capsules != 1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, self.num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.Conv2d(in_channels, num_capsules*out_channels, kernel_size=kernel_size, stride=stride, padding=0) 

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


    def forward(self,x):
        if self.num_in_capsules != 1:
            # print 'x.shape',x.shape
            assert x.shape[2]==x.shape[3]
            # x = x.permute(0,2,3,4,1).contiguous

            # print 'x.shape',x.shape
            width_in = x.shape[2]
            w = width_out = int((width_in-self.kernel_size)/self.stride+1) if self.kernel_size else 1 #5
            # print 'w',w
            outputs = Variable(torch.zeros(self.num_capsules,x.size(0),w,w,self.route_weights.size(-1))).cuda()

            for row in range(w):  # Loop over every pixel of the output
                for col in range(w):
                    col_start = col* self.stride
                    col_end = col*self.stride+self.kernel_size
                    row_start = row* self.stride
                    row_end = row*self.stride+self.kernel_size
                    window = x[:,:,row_start:row_end,col_start:col_end,:].contiguous()
                    window = window.view(1,window.size(0),window.size(1)*window.size(2)*window.size(3),1,window.size(4))

                    priors = torch.matmul(window,self.route_weights[:, None, :, :, :])

                    logits = Variable(torch.zeros(*priors.size())).cuda()
                    for i in range(self.num_iterations):
                        probs = softmax(logits, dim=2)
                        outputs_temp = self.squash((probs * priors).sum(dim=2, keepdim=True))

                        if i != self.num_iterations - 1:
                            delta_logits = (priors * outputs_temp).sum(dim=-1, keepdim=True)
                            logits = logits + delta_logits
                    
                    outputs[:,:,row,col,:] = outputs_temp.squeeze()

            outputs = outputs.permute(1,0,2,3,4).contiguous()
        else:
            outputs = self.capsules(x)
            outputs = outputs.view(outputs.size(0),self.out_channels,self.num_capsules,outputs.size(2),outputs.size(3)).permute(0,2,3,4,1).contiguous()
            # print outputs.shape
            # print raw_input()
            outputs = self.squash(outputs,dim=-1)

        # print outputs.shape
        return outputs

class Dynamic_Capsule_Model(nn.Module):

    def __init__(self,n_classes,conv_layers,caps_layers,r, reconstruct = False):
        super(Dynamic_Capsule_Model, self).__init__()
        print r

        self.reconstruct = reconstruct
        
        # self.num_classes = n_classes
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        # self.primary_capsules = CapsuleLayer(num_capsules=32, num_in_capsules=1, in_channels=256, out_channels=8,
        #                                      kernel_size=9, stride=2)

        # # self.temp = CapsuleLayer(num_capsules=32, num_in_capsules=32, in_channels=8, out_channels=8,
        #                                      # kernel_size=2, stride=1)
        
        # self.digit_capsules = CapsuleLayer(num_capsules=self.num_classes, num_in_capsules=32, in_channels=8,out_channels=16,kernel_size = 6, stride =1)


        self.features = []
        for conv_param in conv_layers:
            self.features.append(nn.Conv2d(in_channels=1, out_channels=conv_param[0],
                                   kernel_size=conv_param[1], stride=conv_param[2]))
            self.features.append(nn.ReLU(True))

        # caps_param <- num_capsules, out_channels,kernel_size, stride

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

        if self.reconstruct:
            self.decoder = nn.Sequential(
                nn.Linear(16 * self.num_classes, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 784),
                nn.Sigmoid()
            )

        
    def forward(self, x, y = None):
        x = self.features(x).squeeze()
        # x = F.relu(self.conv1(x), inplace=True)
        # x = self.primary_capsules(x)
        # # x = self.temp(x)
        # # print x.size()
        # x = self.digit_capsules(x).squeeze()

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)

        if self.reconstruct:
            if y is None:
                # In all batches, get the most active capsule.
                _, max_length_indices = classes.max(dim=1)
                y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=max_length_indices)
            reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
            return classes, reconstructions
        else:
            
            return classes


    def spread_loss(self,x,target,m):
        use_cuda = next(self.parameters()).is_cuda

        b = x.size(0)
        target_t = target.type(torch.LongTensor)
        
        if use_cuda:
            target_t = target_t.cuda()
        
        rows = torch.LongTensor(np.array(range(b)))
        
        if use_cuda:
            rows = rows.cuda()

        a_t = x[rows,target_t]
        a_t_stack = a_t.view(b,1).expand(b,x.size(1)).contiguous() #b,10
        u = m-(a_t_stack-x) #b,10
        u = nn.functional.relu(u)**2
        u[rows,target_t]=0
        loss = torch.sum(u)/b
        
        return loss

    def margin_loss(self,  classes,labels):
      # , images= None, reconstructions=None):
        is_cuda = next(self.parameters()).is_cuda
        # print classes.size()
        if is_cuda:
        # temp = torch.sparse.torch.eye(classes.size(1))
            labels = Variable(torch.sparse.torch.eye(classes.size(1)).cuda().index_select(dim=0, index=labels.data))
        else:
            labels = Variable(torch.sparse.torch.eye(classes.size(1)).index_select(dim=0, index=labels.data))

        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        # if reconstructions is not None:
        #     reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        #     total_loss = (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
        # else:
        total_loss = margin_loss/ labels.size(0)

        return total_loss


class Network:
    def __init__(self,n_classes=10,r=3,input_size=96,conv_layers = None, caps_layers = None):
        if conv_layers is None:
            conv_layers = [[256,9,1]]
        if caps_layers is None:
            caps_layers = [[32,8,9,2],[n_classes,16,6,1]]

        model = Dynamic_Capsule_Model(n_classes,conv_layers,caps_layers,r)

        
        # for idx_m,m in enumerate(model.modules()):
        #     if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
        #         nn.init.xavier_normal(m.weight.data)
        #         nn.init.constant(m.bias.data,0.)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant(m.weight.data,1.)
        #         nn.init.constant(m.bias.data,0.)
                
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
    
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    lr = 0.001
    decay_rate = 0.96
    decay_steps = 469
    min_lr = 1e-6
    optimizer = Adam(network.get_lr_list(0.001))
    exp_lr_scheduler = Exp_Lr_Scheduler(optimizer,0,lr,decay_rate,decay_steps,min_lr)

    batch_size = 128
    test_batch_size = 128

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
            

            classes, reconstructions = model(data, labels)
            # raw_input()
            # print classes.shape, reconstructions.shape
            # else:
            #     classes, reconstructions = model(data)

            loss = model.margin_loss( classes,labels)
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