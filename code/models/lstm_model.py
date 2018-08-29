import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import math
from torch.autograd import Variable

import torch.nn.functional as F
from CapsuleLayer import CapsuleLayer
from caps_rnn_linear import RecurrentCapsuleLayer, softmax

import time

# from dynamic_capsule_layer import RecurrentCapsuleLayer

class LSTM_Model(nn.Module):
    def __init__(self,n_classes,hidden_size, input_size, num_layers, batch_first):
        super(LSTM_Model, self).__init__()
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = batch_first)
        self.classifier = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        # print x.size()
        x = self.classifier(x[:,-1,:])

        # print x.size()
        return x

class Network:
    def __init__(self,n_classes=10,hidden_size=128 ,input_size=1,num_layers = 1, batch_first = True):
        self.model = LSTM_Model(n_classes,hidden_size,input_size,num_layers,batch_first)
        for idx_m,m in enumerate(self.model.modules()):
            if isinstance(m, nn.LSTM):
                for names in m._all_weights:
                    for name in filter(lambda n: 'bias' in n,  names):
                        bias = getattr(m, name)
                        n = bias.size(0)
                        bias.data.fill_(0.)
                        start, end = n//4, n//2
                        bias.data[start:end].fill_(1.)

                    for name in filter(lambda n: 'weight' in n,  names):
                        weight = getattr(m, name)
                        nn.init.xavier_normal(weight.data)        


            if isinstance(m,nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                nn.init.constant(m.bias.data,0.)

        
    def get_lr_list(self, lr):
        
        lr_list= [{'params': self.model.lstm.parameters(), 'lr': lr[0]}]+ [{'params': self.model.classifier.parameters(), 'lr': lr[1]}]
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
    lr = [1e-4,1e-4]
    batch_size = 16
    test_batch_size = 4
    # disp_after = 10
    num_epochs = 1000
    disp_frequency = 1
    hidden_size = 128
    input_size = 1
    num_layers = 1

    network = Network(num_classes,hidden_size = hidden_size, input_size = input_size, num_layers = num_layers)
    model = network.model
    print model

    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    model.cuda()
    
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    
    optimizer = Adam(network.get_lr_list(lr))
        # model.parameters(),lr = lr)
    
    
    transformer = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    train_data = datasets.MNIST('../../data/mnist_data', train=True, transform = transformer, download = True)
    test_data = datasets.MNIST('../../data/mnist_data', train=False, download = True, transform = transformer)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=test_batch_size, shuffle=False)

    
    model.train()
    loss_fun = nn.CrossEntropyLoss().cuda()

    for batch_idx, batch in enumerate(train_loader):
        data = batch[0]
        if input_size == 28:
            data = data.squeeze()
        elif input_size == 1 :
            data = data.view(data.size(0),data.size(1),-1).transpose(1,2).contiguous()
        # print data.size()
        # raw_input()
        labels_a = batch[1]
        
        break

    for epoch_num in range(num_epochs):
        # labels = labels.cuda()
        labels = Variable(labels_a.cuda())
        
        
        classes = model(Variable(data).cuda())

        loss = loss_fun(classes,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # print gt_np,pred_np
        if epoch_num%disp_frequency==0:

            _, preds = torch.max(classes,1)
            gt_np =  labels_a.numpy()
            pred_np =  preds.data.cpu().numpy()
            accuracy = np.sum(gt_np==pred_np)/float(pred_np.size)
            print gt_np, pred_np

            # , torch.norm(model.lstm.grad.data).cpu().numpy()[0],torch.norm(model.classifier.grad.data).cpu().numpy()[0]

            for p in model.parameters():
                print torch.norm(p.grad.data).cpu().numpy()
            print 'Epoch %d, Loss %.2f, Accuracy %.2f' % (epoch_num, loss, accuracy)
            if accuracy==1:
                raw_input()
        
        
        

            
        
             




    

if __name__=='__main__':
    main()