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
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

class RecurrentCapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_in_capsules, in_channels, out_channels, kernel_size, stride,
                 num_iterations=3, w = 1 ):
        
        super(RecurrentCapsuleLayer, self).__init__()

        self.num_route_nodes = num_in_capsules*kernel_size*kernel_size
        self.hidden = None
        # Variable(torch.zeros(self.num_capsules,x.size(0),w,w,self.route_weights.size(-1))).cuda()
        # outputs.permute(1,0,2,3,4).contiguous()

        self.num_in_capsules = num_in_capsules
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels

        self.route_weights = nn.Parameter(torch.randn(num_capsules, self.num_route_nodes, in_channels, out_channels))
        # self.bias = nn.Parameter(torch.Tensor(num_capsules,  out_channels).fill_(0.))

        self.kernel_size_hidden = 1
        self.num_route_nodes_hidden = num_capsules*self.kernel_size_hidden*self.kernel_size_hidden
        self.route_weights_h = nn.Parameter(torch.randn(num_capsules, self.num_route_nodes_hidden, out_channels, out_channels))
        
        self.bias = nn.Parameter(torch.Tensor(num_capsules,  out_channels).fill_(0.))
        self.hidden = Variable(torch.Tensor(4,self.num_capsules,w,w,self.route_weights.size(-1)), requires_grad = False).fill_(0.).cuda()
        self.w = w

        # (j×1×n×m)  tensor and tensor2 is a (k×m×p) tensor, out will be an (j×k×n×p) tensor.

        # self.route_weights = nn.Parameter(torch.randn(num_capsules, self.num_route_nodes, in_channels, out_channels))
        # self.num_route_nodes_hidden = num_capsules*self.kernel_size_hidden*self.kernel_size_hidden
        # self.route_weights_h = nn.Parameter(torch.randn(num_capsules, self.num_route_nodes_hidden, out_channels, out_channels))
        
        # self.bias = nn.Parameter(torch.Tensor(num_capsules,  out_channels).fill_(0.))
        # self.hidden = Variable(torch.Tensor(4,self.num_capsules,w,w,self.route_weights.size(-1)), requires_grad = False).fill_(0.).cuda()
        # self.w = w


    def init_hidden(self):
        self.hidden = Variable(torch.Tensor(4,self.num_capsules,self.w,self.w,self.route_weights.size(-1)), requires_grad = False).fill_(0.).cuda()


    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    # def forward(self,x,hidden):

    # def forward_linear(self,x):

    #     # input - batch size x dimension
    #     # output - batch size x num caps x num caps dims 



    #     t = time.time()

    #     assert x.shape[2]==x.shape[3]
    #     width_in = x.shape[2]
    #     w = width_out = int((width_in-self.kernel_size)/self.stride+1) if self.kernel_size else 1 
    #     print 'w',w
    #     #5
    #     # print 'w',w
    #     t1 = time.time()
    #     outputs = Variable(torch.zeros(self.num_capsules,x.size(0),w,w,self.route_weights.size(-1))).cuda()
    #     print time.time()-t1

    #     # t1 = time.time()
    #     # if self.hidden is None:
    #     #     self.hidden = Variable(torch.zeros(x.size(0),self.num_capsules,w,w,self.route_weights.size(-1))).cuda()
    #     # print time.time()-t1

    #     print 'before loop in forward',time.time()-t

    #     # for row in range(w):  # Loop over every pixel of the output
    #     #     for col in range(w):
    #     row = 0
    #     col = 0
    #     t = time.time()
    #     col_start = col* self.stride
    #     col_end = col*self.stride+self.kernel_size
    #     row_start = row* self.stride
    #     row_end = row*self.stride+self.kernel_size
    #     # print 'in forward in loop'
    #     # print x.size()
    #     # print row_start,row_end,col_start,col_end
    #     window = x[:,:,row_start:row_end,col_start:col_end,:].contiguous()
    #     # print window.size()
        

    #     t1 = time.time()
    #     window = window.view(1,window.size(0),window.size(1)*window.size(2)*window.size(3),1,window.size(4))
    #     print 'window',time.time()-t1
    #     # print window.size()
    #     # print self.route_weights.size()
    #     # print self.route_weights[:, None, :, :, :].size()
    #     t1 = time.time()
    #     priors = torch.matmul(window,self.route_weights[:, None, :, :, :])
    #     print 'priors',time.time()-t1

    #     # print priors.size()
    #     # raw_input()
    #     t1 = time.time()
    #     logits = Variable(torch.zeros(*priors.size())).cuda()
    #     print 'logits',time.time() - t1

    #     print 'priors size',priors.size()
    #     print 'window size',window.size()
    #     print 'self.route_weights',self.route_weights.size()
    #     print 'self.route_weights_h',self.route_weights_h.size()



    #     t1 = time.time()
    #     window_h = self.hidden
    #     # self.hidden[:,:,row_start:row_end,col_start:col_end,:].contiguous()


    #     # print 'window_h.size()',window_h.size()
    #     # print 'self.route_weights_h.size()',self.route_weights_h.size()
    #     # print 'self.route_weights.size()',self.route_weights.size()
    #     window_h = window_h.view(1,window_h.size(0),window_h.size(1)*window_h.size(2)*window_h.size(3),1,window_h.size(4))
    #     priors_h = torch.matmul(window_h,self.route_weights_h[:, None, :, :, :])
    #     print 'bef logits_h',time.time()-t1
    #     print window_h.size()
    #     print priors_h.size()
    #     print self.route_weights_h.size()



    #     logits_h = Variable(torch.zeros(*priors_h.size())).cuda()
        


    #     # print 'window.size()',window.size()
    #     # print 'priors.size()',priors.size()
    #     # print 'logits.size()',logits.size()

    #     # print 'window_h.size()',window_h.size()
    #     # print 'priors_h.size()',priors_h.size()
    #     # print 'logits_h.size()',logits_h.size()
    #     print 'before iter conv loop',time.time()-t

    #     for i in range(self.num_iterations):
    #         t = time.time()

    #         probs = softmax(logits, dim=2)
    #         probs_h = softmax(logits_h, dim=2)
    #         bias = self.bias[:,None,None,None,:]

    #         outputs = self.squash((probs*priors).sum(dim=2, keepdim=True)+ (probs_h*priors_h).sum(dim=2, keepdim=True) + self.bias[:,None,None,None,:])

    #         if i != self.num_iterations - 1:
    #             delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
    #             logits = logits + delta_logits

    #             delta_logits_h = (priors_h * outputs).sum(dim=-1, keepdim=True)
    #             logits_h = logits_h + delta_logits_h

    #         print 'single iteration',time.time()-t

    #             # print 'outputs_temp',outputs_temp.size()
    #             # outputs[:,:,row,col,:] = outputs_temp.squeeze()
    #             # print 'outputs',outputs.size()

    #     t = time.time()

    #     outputs = outputs.permute(1,0,2,3,4).contiguous()
    #     # print torch.min(self.hidden), torch.max(self.hidden)
    #     self.hidden = outputs.clone()
    #     # print torch.min(self.hidden), torch.max(self.hidden)
    #     print 'after conv loop',time.time() - t
    #     return outputs

    def forward(self,x):
        

        t = time.time()

        assert x.shape[2]==x.shape[3]
        width_in = x.shape[2]
        w = width_out = int((width_in-self.kernel_size)/self.stride+1) if self.kernel_size else 1 
        print 'w',w
        #5
        # print 'w',w
        t1 = time.time()
        outputs = Variable(torch.zeros(self.num_capsules,x.size(0),w,w,self.route_weights.size(-1))).cuda()
        print time.time()-t1

        # t1 = time.time()
        # if self.hidden is None:
        #     self.hidden = Variable(torch.zeros(x.size(0),self.num_capsules,w,w,self.route_weights.size(-1))).cuda()
        # print time.time()-t1

        print 'before loop in forward',time.time()-t

        # for row in range(w):  # Loop over every pixel of the output
        #     for col in range(w):
        row = 0
        col = 0
        t = time.time()
        col_start = col* self.stride
        col_end = col*self.stride+self.kernel_size
        row_start = row* self.stride
        row_end = row*self.stride+self.kernel_size
        # print 'in forward in loop'
        # print x.size()
        # print row_start,row_end,col_start,col_end
        window = x[:,:,row_start:row_end,col_start:col_end,:].contiguous()
        # print window.size()
        

        t1 = time.time()
        window = window.view(1,window.size(0),window.size(1)*window.size(2)*window.size(3),1,window.size(4))
        print 'window',time.time()-t1
        # print window.size()
        # print self.route_weights.size()
        # print self.route_weights[:, None, :, :, :].size()
        t1 = time.time()
        priors = torch.matmul(window,self.route_weights[:, None, :, :, :])
        print 'priors',time.time()-t1

        # print priors.size()
        # raw_input()
        t1 = time.time()
        logits = Variable(torch.zeros(*priors.size())).cuda()
        print 'logits',time.time() - t1

        print 'priors size',priors.size()
        print 'window size',window.size()
        print 'self.route_weights',self.route_weights.size()
        print 'self.route_weights_h',self.route_weights_h.size()



        t1 = time.time()
        window_h = self.hidden
        # self.hidden[:,:,row_start:row_end,col_start:col_end,:].contiguous()


        # print 'window_h.size()',window_h.size()
        # print 'self.route_weights_h.size()',self.route_weights_h.size()
        # print 'self.route_weights.size()',self.route_weights.size()
        window_h = window_h.view(1,window_h.size(0),window_h.size(1)*window_h.size(2)*window_h.size(3),1,window_h.size(4))
        priors_h = torch.matmul(window_h,self.route_weights_h[:, None, :, :, :])
        print 'bef logits_h',time.time()-t1
        print window_h.size()
        print priors_h.size()
        print self.route_weights_h.size()



        logits_h = Variable(torch.zeros(*priors_h.size())).cuda()
        


        # print 'window.size()',window.size()
        # print 'priors.size()',priors.size()
        # print 'logits.size()',logits.size()

        # print 'window_h.size()',window_h.size()
        # print 'priors_h.size()',priors_h.size()
        # print 'logits_h.size()',logits_h.size()
        print 'before iter conv loop',time.time()-t

        for i in range(self.num_iterations):
            t = time.time()

            probs = softmax(logits, dim=2)
            probs_h = softmax(logits_h, dim=2)
            bias = self.bias[:,None,None,None,:]

            outputs = self.squash((probs*priors).sum(dim=2, keepdim=True)+ (probs_h*priors_h).sum(dim=2, keepdim=True) + self.bias[:,None,None,None,:])

            if i != self.num_iterations - 1:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits

                delta_logits_h = (priors_h * outputs).sum(dim=-1, keepdim=True)
                logits_h = logits_h + delta_logits_h

            print 'single iteration',time.time()-t

                # print 'outputs_temp',outputs_temp.size()
                # outputs[:,:,row,col,:] = outputs_temp.squeeze()
                # print 'outputs',outputs.size()

        t = time.time()

        outputs = outputs.permute(1,0,2,3,4).contiguous()
        # print torch.min(self.hidden), torch.max(self.hidden)
        self.hidden = outputs.clone()
        # print torch.min(self.hidden), torch.max(self.hidden)
        print 'after conv loop',time.time() - t
        return outputs