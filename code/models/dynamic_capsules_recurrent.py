import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import math
from torch.autograd import Variable

import torch.nn.functional as F
from CapsuleLayer import CapsuleLayer
from caps_indrnn_linear import RecurrentCapsuleLayer, softmax

import time

# from dynamic_capsule_layer import RecurrentCapsuleLayer

class Recurrent_Dynamic_Capsule_Model(nn.Module):

    def __init__(self,n_classes,conv_layers,caps_layers,r, bs, reconstruct = False,class_weights=None):
        super(Recurrent_Dynamic_Capsule_Model, self).__init__()
        print r
        self.num_classes = n_classes
        self.recurrences = []
        for idx_caps_param,caps_param in enumerate(caps_layers):
            num_capsules, out_channels,num_in_capsules, in_channels = caps_param
            print num_capsules, num_in_capsules, in_channels, out_channels

            self.recurrences.append(RecurrentCapsuleLayer(num_capsules, num_in_capsules, in_channels, out_channels,  num_iterations=r, batch_size = bs))

        self.recurrences = nn.Sequential(*self.recurrences)
        
    def forward(self, data, y = None,return_caps = False):
        
        # t = time.time()
        for row_num in range(data.size(2)):
            x = Variable(data[:,:,row_num,:].cuda())
            x = self.recurrences(x)

        # for row_num in range(data.size(2)):
        #     for col_num in range(data.size(3)):
        #         x = Variable(data[:,:,row_num,col_num].cuda())
        #         x = torch.unsqueeze(x,2)
        #         # print x.size()
        #         # raw_input()
        #         x = self.recurrences(x)
        
        classes = (x ** 2).sum(dim=-1) ** 0.5
        # print 'time',time.time()-t

        return classes

    
    def clear_hidden(self):
        for r in self.recurrences:
            r.init_hidden()



class Network:
    def __init__(self,n_classes=10,r=3 ,conv_layers = None, caps_layers = None,reconstruct=False,bs = 1):
        if conv_layers is None:
            conv_layers = [[256,9,1]]
        if caps_layers is None:
            caps_layers = [[10,8,1,1]]

        model = Recurrent_Dynamic_Capsule_Model(n_classes,conv_layers,caps_layers,r,reconstruct=reconstruct,bs = bs)

        for l in model.recurrences:
            nn.init.xavier_normal(l.route_weights.data)
            nn.init.xavier_normal(l.route_weights_h.data)
            nn.init.constant(l.bias.data,0)
            # nn.init.xavier_normal(l.route_weights.data)

        # for idx_m,m in enumerate(model.modules()):
        #     if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
        #         # print m
        #         nn.init.xavier_normal(m.weight.data,std=5e-2)
        #         nn.init.constant(m.bias.data,0.)
        #     elif isinstance(m, RecurrentCapsuleLayer):
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
        
        lr_list= [{'params': self.model.recurrences.parameters(), 'lr': lr[0]}]
        # if self.model.reconstruct:
        #     lr_list= lr_list + [{'params': self.model.decoder.parameters(), 'lr': lr[1]}]
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
    batch_size = 2
    test_batch_size = 4
    # caps_layers = [[32,8,1,28],[10,1,32,8]]
    caps_layers = [[10,8,1,28]]
    # caps_layers = [[10,8,1,1]]
    lr = [1e-2]
    disp_after = 1
    num_epochs = 10000
    


    network = Network(num_classes, bs = batch_size, caps_layers = caps_layers)
    model = network.model
    model.cuda()
    model.train()
    loss_fun = nn.CrossEntropyLoss().cuda()
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(network.get_lr_list(lr))

    transformer = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    train_data = datasets.MNIST('../../data/mnist_data', train=True, transform = transformer, download = True)
    test_data = datasets.MNIST('../../data/mnist_data', train=False, download = True, transform = transformer)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=test_batch_size, shuffle=False)

    
    
    for batch_idx, batch in enumerate(train_loader):
        data = batch[0]
        labels_a = batch[1]
        
        break

    for epoch_num in range(num_epochs):
        # labels = labels.cuda()
        labels = Variable(labels_a.cuda())
        
        t = time.time()
        classes = model(data)
        t = time.time() - t

        _, preds = torch.max(classes,1)

        
        
        # loss,_,_ = model.margin_loss( classes,labels)
        loss = loss_fun(classes,labels)

        # if batch_idx % disp_after ==0:  
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch_num, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        model.clear_hidden()

        _, preds = torch.max(classes,1)
        gt_np =  labels_a.numpy()
        pred_np =  preds.data.cpu().numpy()
        accuracy = np.sum(gt_np==pred_np)/float(pred_np.size)
        
        if epoch_num%disp_after==0 or accuracy==1:
            print t
            
            print gt_np, pred_np

            
            for p in model.parameters():
                print torch.norm(p.grad.data).cpu().numpy()
            print 'Epoch %d, Loss %.2f, Accuracy %.2f' % (epoch_num, loss, accuracy)
            if accuracy==1:
                raw_input()

            # print labels_a
            # print preds
            # print classes

            # print 'route_weights.grad',torch.norm(model.recurrences[0].route_weights.grad).data
            # print 'route_weights_h.grad',torch.norm(model.recurrences[0].route_weights_h.grad).data

            
        
             




    

if __name__=='__main__':
    main()