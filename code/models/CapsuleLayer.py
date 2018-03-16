import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import math
from torch.autograd import Variable

import torch.nn.functional as F

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
            self.bias = nn.Parameter(torch.Tensor(num_capsules,  out_channels).fill_(0.))
        else:
            self.route_weights=False
            self.capsules = nn.Conv2d(in_channels, num_capsules*out_channels, kernel_size=kernel_size, stride=stride, padding=0) 

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


    def forward_intrusive(self,x):
        probs_all = []
        if self.num_in_capsules != 1:
            # print 'x.shape',x.shape
            assert x.shape[2]==x.shape[3]
            # x = x.permute(0,4,1,2,3).contiguous()

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
                    # print window.size()
                    priors = torch.matmul(window,self.route_weights[:, None, :, :, :])
                    # print 'priors.size()',priors.size()
                    # raw_input()

                    # window = x[:,:,:,row_start:row_end,col_start:col_end]
                    # print window.size()
                    # window = window.view(window.size(0),window.size(1),window.size(2)*window.size(3)*window.size(4),1)
                    # print window.size()
                    # window = window.permute(0,1,3,2).contiguous()

                    # priors = torch.matmul(window,self.route_weights[:, None, :, :, :])

                    # print x.size()
                    logits = Variable(torch.zeros(*priors.size())).cuda()
                    for i in range(self.num_iterations):

                        probs = softmax(logits, dim=2)
                        # print 'iter',i,probs.size(),torch.min(probs),torch.max(probs)
                        bias = self.bias[:,None,None,None,:]
                        # print 'bias.size()',bias.size()
                        # t = 
                        # print 't.size()',t.size()
                        # raw_input()
                        outputs_temp = self.squash((probs * priors).sum(dim=2, keepdim=True)+self.bias[:,None,None,None,:])

                        if i != self.num_iterations - 1:
                            delta_logits = (priors * outputs_temp).sum(dim=-1, keepdim=True)
                            logits = logits + delta_logits
                        if i>0:
                            probs_all.append(probs.data.cpu().numpy())
                    # raw_input()
                    outputs[:,:,row,col,:] = outputs_temp.squeeze()

            outputs = outputs.permute(1,0,2,3,4).contiguous()
        else:
            outputs = self.capsules(x)
            outputs = outputs.view(outputs.size(0),self.out_channels,self.num_capsules,outputs.size(2),outputs.size(3)).permute(0,2,3,4,1).contiguous()
            # print outputs.shape
            # print raw_input()
            outputs = self.squash(outputs,dim=-1)

        # print outputs.shape
        return outputs,probs_all


    def forward(self,x):
        if self.num_in_capsules != 1:
            # print 'x.shape',x.shape
            assert x.shape[2]==x.shape[3]
            # x = x.permute(0,4,1,2,3).contiguous()

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
                    # print window.size()
                    priors = torch.matmul(window,self.route_weights[:, None, :, :, :])
                    # print 'priors.size()',priors.size()
                    # raw_input()

                    # window = x[:,:,:,row_start:row_end,col_start:col_end]
                    # print window.size()
                    # window = window.view(window.size(0),window.size(1),window.size(2)*window.size(3)*window.size(4),1)
                    # print window.size()
                    # window = window.permute(0,1,3,2).contiguous()

                    # priors = torch.matmul(window,self.route_weights[:, None, :, :, :])

                    # print x.size()
                    logits = Variable(torch.zeros(*priors.size())).cuda()
                    for i in range(self.num_iterations):
                        probs = softmax(logits, dim=2)
                        # print 'iter',i,probs.size(),torch.min(probs),torch.max(probs)
                        bias = self.bias[:,None,None,None,:]
                        # print 'bias.size()',bias.size()
                        # t = 
                        # print 't.size()',t.size()
                        # raw_input()
                        outputs_temp = self.squash((probs * priors).sum(dim=2, keepdim=True)+self.bias[:,None,None,None,:])

                        if i != self.num_iterations - 1:
                            delta_logits = (priors * outputs_temp).sum(dim=-1, keepdim=True)
                            logits = logits + delta_logits
                    # raw_input()
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