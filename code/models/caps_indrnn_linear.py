import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import math
from torch.autograd import Variable

import torch.nn.functional as F


import torch.nn as nn
import time

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()

#         self.hidden_size = hidden_size

#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden), 1)
#         hidden = self.i2h(combined)
#         output = self.i2o(combined)
#         output = self.softmax(output)
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, self.hidden_size)

# n_hidden = 128
# rnn = RNN(n_letters, n_hidden, n_categories)




def softmax(input, dim=1):
    # print 'in softmax'
    # print dim
    # print input.size()
    # transposed_input = input.transpose(dim, len(input.size()) - 1)
    # print transposed_input.size()
    # softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    # print softmaxed_output.size()
    # to_return = softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)
    # print to_return.size()
    # print 'end of softmax'

    # check_out = F.softmax(input,dim)
    # print check_out.size()
    # raw_input()
    return F.softmax(input,dim)


class RecurrentCapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_in_capsules, in_channels, out_channels, num_iterations=3, w = 1 ,batch_size = 2):
        
        super(RecurrentCapsuleLayer, self).__init__()


        self.num_in_capsules = num_in_capsules
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        # self.kernel_size = kernel_size
        # self.stride = stride
        self.out_channels = out_channels

        # print 'self.num_in_capsules',self.num_in_capsules
        # print 'self.num_iterations',self.num_iterations
        # print 'self.num_capsules',self.num_capsules
        # print 'self.out_channels',self.out_channels

        self.route_weights = nn.Parameter(torch.randn(num_capsules, self.num_in_capsules, in_channels, out_channels))
        
        # self.num_route_nodes_hidden = num_capsules*self.kernel_size_hidden*self.kernel_size_hidden
        self.route_weights_h = nn.Parameter(torch.randn(self.num_capsules, self.num_capsules, out_channels, out_channels))
        
        self.bias = nn.Parameter(torch.Tensor(num_capsules, out_channels).fill_(0.))

        self.batch_size = batch_size
        self.hidden = Variable(torch.Tensor(self.batch_size,self.num_capsules,out_channels), requires_grad = False).fill_(0.).cuda()


    def init_hidden(self):
        self.hidden = Variable(torch.Tensor(self.batch_size,self.num_capsules,self.out_channels), requires_grad = False).fill_(0.).cuda()


    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    
    def forward(self,x):
        
        

        # input bs x num_caps x caps_dim
        print 'x before', x.size()
        x = x.view(1,x.size(0),x.size(1),1,x.size(2)) # bs x 1 x num_caps x caps_dim
        print 'x before', x.size()
        print 'route_weights', self.route_weights.size()
        priors = torch.matmul(x,self.route_weights[ :, None, :, :, :])
        print 'priors', priors.size()
        

        logits = Variable(torch.zeros(*priors.size())).cuda()

        print 'hidden before', self.hidden.size()
        self.hidden = self.hidden.view(1,self.hidden.size(0),self.hidden.size(1),1,self.hidden.size(2))
        print 'hidden before', self.hidden.size()
        print 'route_weights_h', self.route_weights_h.size()
        priors_h = torch.matmul(self.hidden,self.route_weights_h[ :, None, :, :, :])
        print 'priors_h', priors_h.size()
        raw_input()

        logits_h = Variable(torch.zeros(*priors_h.size())).cuda()
        # print logits.size()
        # print logits_h.size()
        # raw_input()
        for i in range(self.num_iterations):
            # print 'i',i
            probs = softmax(logits, dim=2)
            probs_h = softmax(logits_h, dim=2)
            # print priors_h.size()
            # print probs_h.size()

            # print (probs*priors).sum(dim=2, keepdim=True).size()
            # print (probs_h*priors_h).sum(dim=2, keepdim=True).size()
            # print self.bias[:,None,None,None,:].size()
            # raw_input()

            outputs = self.squash((probs*priors).sum(dim=2, keepdim=True)+(probs_h*priors_h).sum(dim=2, keepdim=True)+self.bias[:,None,None,None,:])
            
            if i != self.num_iterations - 1:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits

                delta_logits_h = (priors_h * outputs).sum(dim=-1, keepdim=True)
                logits_h = logits_h + delta_logits_h

        outputs = outputs.permute(1,0,2,3,4).contiguous()
        outputs = outputs.view(outputs.size(0),outputs.size(1),outputs.size(4))
        self.hidden = outputs.clone()

        return outputs

