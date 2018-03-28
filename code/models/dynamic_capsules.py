import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import math
from torch.autograd import Variable

import torch.nn.functional as F
from CapsuleLayer import CapsuleLayer, softmax

# from dynamic_capsule_layer import CapsuleLayer


class Dynamic_Capsule_Model_Super(nn.Module):

    def __init__(self):
        super(Dynamic_Capsule_Model_Super, self).__init__()
       
    def just_reconstruct(self,x,y=None):

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=max_length_indices)
        else:
            y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=y)
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(reconstructions.size(0),1,int(math.sqrt(reconstructions.size(1))),int(math.sqrt(reconstructions.size(1))))
        if hasattr(self, 'upsampler'):
            reconstructions = self.upsampler(reconstructions)

        return reconstructions

    def forward(self, data, y = None,return_caps = False):
        # print 'IN FORWARD',self.reconstruct,data.size(),y.size(),
        
        x = self.features(data).squeeze()
        # print x.size()

        # x = F.relu(self.conv1(x), inplace=True)
        # x = self.primary_capsules(x)
        # # x = self.temp(x)
        # # print x.size()
        # x = self.digit_capsules(x).squeeze()

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)

        if self.reconstruct:
            # print 'y',y.size()
            if y is None:
                # In all batches, get the most active capsule.
                _, max_length_indices = classes.max(dim=1)
                y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=max_length_indices)
            else:
                y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=y)
            
            # print y.size()
            # print x.size()
            # raw_input()
            reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
            reconstructions = reconstructions.view(reconstructions.size(0),1,int(math.sqrt(reconstructions.size(1))),int(math.sqrt(reconstructions.size(1))))
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


    # def spread_loss(self,x,target,m):
    #     use_cuda = next(self.parameters()).is_cuda

    #     b = x.size(0)
        
    #     rows = torch.LongTensor(np.array(range(b)))
        
    #     if use_cuda:
    #         rows = rows.cuda()
    #     a_t = torch.diag(torch.index_select(x,1,target))
        

    #     a_t_stack = a_t.view(b,1).expand(b,x.size(1)).contiguous() #b,10

    #     u = m-(a_t_stack-x) #b,10
    #     u = nn.functional.relu(u)**2
    #     u[rows,target]=0
    #     loss = torch.sum(u)/b
        
    #     return loss



    def margin_loss(self,  classes,labels):
        if self.reconstruct:
            images = classes[2]
            reconstructions = classes[1]
            classes = classes[0]

      # , images= None, reconstructions=None):
        is_cuda = next(self.parameters()).is_cuda
        batch_size = labels.size(0)

        # print classes.size()
        if is_cuda:
        # temp = torch.sparse.torch.eye(classes.size(1))
            labels = Variable(torch.sparse.torch.eye(classes.size(1)).cuda().index_select(dim=0, index=labels.data))
        else:
            labels = Variable(torch.sparse.torch.eye(classes.size(1)).index_select(dim=0, index=labels.data))

        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2


        margin_loss = labels * left + 0.5 * (1. - labels) * right
        # print margin_loss[0]

        if hasattr(self, 'class_weights') and self.class_weights is not None:
            # print margin_loss.size(),type(margin_loss)
            
            # print class_weights.size()
            if is_cuda:
                class_weights = torch.autograd.Variable(self.class_weights.cuda())
            else:
                class_weights = torch.autograd.Variable(self.class_weights)

            margin_loss = margin_loss*class_weights

        # print margin_loss[0]
        # raw_input()
        margin_loss = margin_loss.sum()
        margin_loss = margin_loss/ batch_size

        if self.reconstruct:
            reconstruction_loss = self.reconstruction_loss(reconstructions, images)
            reconstruction_loss = (0.0000001 * reconstruction_loss)/batch_size

            if hasattr(self, 'loss_weights') and self.loss_weights is not None:
                reconstruction_loss = self.loss_weights[1]*reconstruction_loss
                margin_loss = self.loss_weights[0]*margin_loss
            
            total_loss = margin_loss + reconstruction_loss
            return total_loss, margin_loss, reconstruction_loss
        else:
            total_loss = margin_loss
            return total_loss, margin_loss, margin_loss
        

    def margin_loss_multi(self,  classes, labels, labels_au, bin_au):
        if self.reconstruct:
            images = classes[3]
            reconstructions = classes[2]
            classes_au = classes[1]
            classes = classes[0]
        else:
            classes_au = classes[1]
            classes = classes[0]

      # , images= None, reconstructions=None):
        is_cuda = next(self.parameters()).is_cuda
        batch_size = labels.size(0)

        # print classes.size()
        if is_cuda:
        # temp = torch.sparse.torch.eye(classes.size(1))
            labels = Variable(torch.sparse.torch.eye(classes.size(1)).cuda().index_select(dim=0, index=labels.data))
        else:
            labels = Variable(torch.sparse.torch.eye(classes.size(1)).index_select(dim=0, index=labels.data))

        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2


        margin_loss = labels * left + 0.5 * (1. - labels) * right
        # print margin_loss[0]

        classes_au = classes_au[:,:labels_au.size(1)]
        

        left_au = F.relu(0.9 - classes_au, inplace=True) ** 2
        right_au = F.relu(classes_au - 0.1, inplace=True) ** 2
        margin_loss_au = labels_au * left_au + 0.5 * (1. - labels_au) * right_au
        
        # print margin_loss_au.size(),bin_au.size()
        # print margin_loss_au
        # print bin_au
        # print labels_au
        
        margin_loss_au = margin_loss_au * bin_au.view(bin_au.size(0),1)
        # print margin_loss_au.size()



        if hasattr(self, 'class_weights') and self.class_weights is not None:
            # print margin_loss.size(),type(margin_loss)
            
            # print class_weights.size()
            if is_cuda:
                class_weights = torch.autograd.Variable(self.class_weights.cuda())
            else:
                class_weights = torch.autograd.Variable(self.class_weights)

            margin_loss = margin_loss*class_weights

        if hasattr(self, 'class_weights_au') and self.class_weights_au is not None:
            # print margin_loss.size(),type(margin_loss)
            
            # print class_weights.size()
            if is_cuda:
                class_weights_au = torch.autograd.Variable(self.class_weights_au.cuda())
            else:
                class_weights_au = torch.autograd.Variable(self.class_weights_au)

            margin_loss_au = margin_loss_au*class_weights_au



        # print margin_loss[0]
        # raw_input()
        margin_loss = margin_loss.sum()
        margin_loss = margin_loss/ batch_size

        margin_loss_au = margin_loss_au.sum()
        margin_loss_au = margin_loss_au/ max(torch.sum(labels_au.data),1)        

        if self.reconstruct:
            reconstruction_loss = self.reconstruction_loss(reconstructions, images)
            reconstruction_loss = (0.0000001 * reconstruction_loss)/batch_size
            if self.loss_weights is not None:
                total_loss = self.loss_weights[0]*margin_loss + self.loss_weights[1]*margin_loss_au + self.loss_weights[2]*reconstruction_loss
            else:    
                total_loss = margin_loss + margin_loss_au + reconstruction_loss
            return total_loss, margin_loss, margin_loss_au, reconstruction_loss
        else:
            total_loss = margin_loss + margin_loss_au
            return total_loss, margin_loss, margin_loss_au


class Dynamic_Capsule_Model(Dynamic_Capsule_Model_Super):

    def __init__(self,n_classes,conv_layers,caps_layers,r, reconstruct = False,class_weights=None):
        super(Dynamic_Capsule_Model, self).__init__()
        print r
        self.class_weights = class_weights
        self.num_classes = n_classes
        self.reconstruct = reconstruct
        if self.reconstruct:
            self.reconstruction_loss = nn.MSELoss(size_average=False)
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
                nn.Linear(out_channels * self.num_classes, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 784),
                # nn.Tanh()
            )

    


class Network:
    def __init__(self,n_classes=10,r=3,input_size=96,conv_layers = None, caps_layers = None,reconstruct=False):
        if conv_layers is None:
            conv_layers = [[256,9,1]]
        if caps_layers is None:
            caps_layers = [[32,8,9,2],[n_classes,16,6,1]]

        model = Dynamic_Capsule_Model(n_classes,conv_layers,caps_layers,r,reconstruct=reconstruct)

        
        # for idx_m,m in enumerate(model.modules()):
        #     if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
        #         # print m
        #         nn.init.xavier_normal(m.weight.data,std=5e-2)
        #         nn.init.constant(m.bias.data,0.)
        #     elif isinstance(m, CapsuleLayer):
        #         if m.num_in_capsules==1:
        #             nn.init.normal(m.capsules.weight.data,std=0.1)
        #             nn.init.constant(m.capsules.bias.data,0.)
        #         else:
        #             nn.init.normal(m.route_weights.data, mean=0, std=0.1)
                    
                # nn.init.normal(m.weight.data,std=0.1)
        #         nn.init.constant(m.weight.data,1.)
        #         nn.init.constant(m.bias.data,0.)
                
        self.model = model

    def get_lr_list(self, lr):
        
        lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]
        if self.model.reconstruct:
            lr_list= lr_list + [{'params': self.model.decoder.parameters(), 'lr': lr[1]}]
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