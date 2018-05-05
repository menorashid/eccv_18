import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import math
from torch.autograd import Variable
import random
import torch.nn.functional as F

def softmax(input, dim=1,arr_to_keep = None):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    # print 'transposed_input.size()',transposed_input.size()
    x = transposed_input.contiguous().view(-1, transposed_input.size(-1))
    # print 'x.size()',x.size()
    if arr_to_keep is not None:
        # print 'DROPPING softmax'
        softmaxed_output_temp = F.softmax(x[:,arr_to_keep])
        softmaxed_output = x.clone()*0
        softmaxed_output.index_copy_(1, Variable(torch.LongTensor(arr_to_keep).cuda()).detach(), softmaxed_output_temp)
    else:
        softmaxed_output = F.softmax(x)
    # print torch.min(softmaxed_output),torch.max(softmaxed_output),softmaxed_output.size()

    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_in_capsules, in_channels, out_channels, kernel_size, stride,
                 num_iterations=3, dropout = 0):
        
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_in_capsules*kernel_size*kernel_size
        self.num_in_capsules = num_in_capsules
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels
        self.dropout = dropout



        if self.num_in_capsules != 1:

            self.route_weights = nn.Parameter(torch.randn(num_capsules, self.num_route_nodes, in_channels, out_channels))
            self.bias = nn.Parameter(torch.Tensor(num_capsules,  out_channels).fill_(0.))
        else:
            assert self.dropout==0
            
            self.route_weights=False
            self.capsules = nn.Conv2d(in_channels, num_capsules*out_channels, kernel_size=kernel_size, stride=stride, padding=0) 


    def squash(self, tensor, dim=-1):
        # print 'in squash'
        # print torch.min(tensor).data[0],torch.max(tensor).data[0]
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        # print torch.min(squared_norm).data[0],torch.max(squared_norm).data[0]
        scale = squared_norm / (1 + squared_norm)
        out = scale * tensor / torch.sqrt(squared_norm)
        
        # temp = out!=out
        # temp = temp.view(-1).long().sum()
        # print temp
        
        out[(out!=out).detach()]=0.

        # print torch.min(scale).data[0],torch.max(scale).data[0]
        # print torch.min(torch.sqrt(squared_norm)).data[0],torch.max(torch.sqrt(squared_norm)).data[0]
        # print torch.min(out).data[0],torch.max(out).data[0]
        # print '----'
        # # ,scale.size()
        return out
        # scale * tensor / torch.sqrt(squared_norm)


    def forward_intrusive(self,x):
        probs_all = []
        if self.num_in_capsules != 1:
            # print 'x.shape',x.shape
            assert x.shape[2]==x.shape[3]
            # x = x.permute(0,4,1,2,3).contiguous()

            # print 'x.shape',x.size()
            # print self.route_weights.size(-1)
            width_in = x.shape[2]
            w = width_out = int((width_in-self.kernel_size)/self.stride+1) if self.kernel_size else 1 #5
            # print 'w',w
            outputs = Variable(torch.zeros(self.num_capsules,x.size(0),w,w,self.route_weights.size(-1))).cuda()

            num_in_routes = self.kernel_size*self.kernel_size*self.num_in_capsules
            arr_to_keep = None
            print 'TRAINING',self.training
            if self.training and self.dropout>0:
                arr_total = list(range(num_in_routes))
                num_to_keep = np.clip(round(len(arr_total)*(1-self.dropout)),1,len(arr_total))
                # num_to_keep = len(arr_to_keep)-num_to_drop
                if num_to_keep<len(arr_total):
                    num_to_keep = int(num_to_keep)
                    arr_to_keep = random.sample(arr_total,num_to_keep)
                    arr_to_keep.sort()
                    arr_to_drop = [val for val in arr_total if val not in arr_to_keep]
                    arr_to_drop.sort()
                    # print len(arr_to_drop)
                    # print len(arr_to_keep)

            for row in range(w):  # Loop over every pixel of the output
                for col in range(w):
                    col_start = col* self.stride
                    col_end = col*self.stride+self.kernel_size
                    row_start = row* self.stride
                    row_end = row*self.stride+self.kernel_size
                    window = x[:,:,row_start:row_end,col_start:col_end,:].contiguous()
                    # print 'windos.size()',window.size()
                    window = window.view(1,window.size(0),window.size(1)*window.size(2)*window.size(3),1,window.size(4))
                    # print 'window.size()',window.size()

                    if arr_to_keep is not None:
                        indexer = Variable(torch.LongTensor(arr_to_drop).cuda()).detach()
                        window = window.index_fill(2,indexer,0)
                        

                    priors = torch.matmul(window,self.route_weights[:, None, :, :, :])
                    logits = Variable(torch.zeros(*priors.size())).cuda()
                    
                    for i in range(self.num_iterations):

                        probs = softmax(logits, dim=2, arr_to_keep=  arr_to_keep)
                        mulled = probs * priors
                    
                        outputs_temp = self.squash((mulled).sum(dim=2, keepdim=True)+self.bias[:,None,None,None,:])

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
            outputs = self.squash(outputs,dim=-1)

        return outputs,probs_all


    def forward(self,x):
        
        forward_debug = False

        if self.num_in_capsules != 1:

            # print 'hello',self.num_in_capsules,self.num_route_nodes,self.dropout,self.route_weights

            # print 'x.shape',x.shape
            assert x.shape[2]==x.shape[3]
            # x = x.permute(0,4,1,2,3).contiguous()

            # print 'x.shape',x.shape
            width_in = x.shape[2]
            w = width_out = int((width_in-self.kernel_size)/self.stride+1) if self.kernel_size else 1 #5
            # print 'w',w
            outputs = Variable(torch.zeros(self.num_capsules,x.size(0),w,w,self.route_weights.size(-1))).cuda()

            num_in_routes = self.kernel_size*self.kernel_size*self.num_in_capsules
            arr_to_keep = None
            if self.training and self.dropout>0:
                arr_total = list(range(num_in_routes))
                num_to_keep = np.clip(round(len(arr_total)*(1-self.dropout)),1,len(arr_total))
                # num_to_keep = len(arr_to_keep)-num_to_drop
                if num_to_keep<len(arr_total):
                    num_to_keep = int(num_to_keep)
                    arr_to_keep = random.sample(arr_total,num_to_keep)
                    arr_to_keep.sort()
                    arr_to_drop = [val for val in arr_total if val not in arr_to_keep]
                    arr_to_drop.sort()
                    if forward_debug:
                        print 'len(arr_to_drop)',len(arr_to_drop)
                        print 'len(arr_to_keep)',len(arr_to_keep)
            if forward_debug:    
                print 'self.training',self.training
                print 'num_in_routes',num_in_routes
                print 'outputs.size',outputs.size()
                print 'self.route_weights.size()',self.route_weights.size()
                print 'self.bias.size()',self.bias.size()

            for row in range(w):  # Loop over every pixel of the output
                for col in range(w):
                    col_start = col* self.stride
                    col_end = col*self.stride+self.kernel_size
                    row_start = row* self.stride
                    row_end = row*self.stride+self.kernel_size
                    window = x[:,:,row_start:row_end,col_start:col_end,:].contiguous()
                    # print 'windos.size()',window.size()
                    window = window.view(1,window.size(0),window.size(1)*window.size(2)*window.size(3),1,window.size(4))
                    # print 'window.size()',window.size()

                    if arr_to_keep is not None:
                        # print 'DROPPING'
                        # print window[0,0,torch.LongTensor(arr_to_keep).cuda(),0,0]
                        # print torch.LongTensor(arr_to_keep).cuda().size()
                        # print arr_to_keep

                        indexer = Variable(torch.LongTensor(arr_to_drop).cuda()).detach()
                        # print indexer
                        # print window[0,0,arr_to_keep[0],0,0]
                        # print window[0,0,0,0,0]
                        
                        window = window.index_fill(2,indexer,0)
                        # print window[0,0,arr_to_keep[0],0,0]
                        # print window[0,0,0,0,0]
                        # raw_input()

                    # print window.size(),self.route_weights.size(),self.num_in_capsules
                    priors = torch.matmul(window,self.route_weights[:, None, :, :, :])
                    # *0
                    # print 'priors.size()',priors.size()
                    # raw_input()

                    # window = x[:,:,:,row_start:row_end,col_start:col_end]
                    # print window.size()
                    # window = window.view(window.size(0),window.size(1),window.size(2)*window.size(3)*window.size(4),1)
                    # print window.size()
                    # window = window.permute(0,1,3,2).contiguous()

                    # priors = torch.matmul(window,self.route_weights[:, None, :, :, :])

                    # print x.size()
                    logits = Variable(torch.zeros(*priors.size())).cuda().detach()
                    # logits = Variable(torch.zeros(priors.size(0),priors.size(1),len(arr_to_keep),priors.size(3),priors.size(4))).cuda()
                    # print 'logits.size()',logits.size()
                    if forward_debug:
                        print 'priors.size()',priors.size()
                        print 'priors.data[0,0,arr_to_keep[0],0,0]',priors.data[0,0,arr_to_keep[0],0,0]
                        print 'priors.data[0,0,arr_to_drop[0],0,0]',priors.data[0,0,arr_to_drop[0],0,0]

                    for i in range(self.num_iterations):
                        
                        probs = softmax(logits, dim=2, arr_to_keep=  arr_to_keep)
                        if forward_debug:
                            print 'iter',i
                            print 'probs',probs.size()
                            print 'probs.data[0,0,arr_to_keep[0],0,0]',probs.data[0,0,arr_to_keep[0],0,0]
                            print 'probs.data[0,0,arr_to_drop[0],0,0]',probs.data[0,0,arr_to_drop[0],0,0]

                        
                        
                        # expand probs


                        # print 'iter',i,probs.size(),torch.min(probs),torch.max(probs)
                        
                        # print 'bias.size()',bias.size()
                        # t = 
                        # print 't.size()',t.size()
                        # raw_input()
                        mulled = probs * priors
                        if forward_debug:
                            print 'mulled',mulled.size()
                            print 'mulled.data[0,0,arr_to_keep[0],0,0]',mulled.data[0,0,arr_to_keep[0],0,0]
                            print 'mulled.data[0,0,arr_to_drop[0],0,0]',mulled.data[0,0,arr_to_drop[0],0,0]
 
                        outputs_temp = self.squash((mulled).sum(dim=2, keepdim=True)+self.bias[:,None,None,None,:])
                        if forward_debug:
                            print 'outputs_temp.size()',outputs_temp.size()
                        if i != self.num_iterations - 1:
                            delta_logits = (priors * outputs_temp).sum(dim=-1, keepdim=True)
                            logits = logits + delta_logits

                            if forward_debug:
                                print 'logits.size()',logits.size()
                                print 'logits.data[0,0,arr_to_keep[0],0,0]',logits.data[0,0,arr_to_keep[0],0,0]
                                print 'logits.data[0,0,arr_to_drop[0],0,0]',logits.data[0,0,arr_to_drop[0],0,0]
 


                    outputs[:,:,row,col,:] = outputs_temp.squeeze()

            outputs = outputs.permute(1,0,2,3,4).contiguous()
            if forward_debug:
                raw_input()
        else:
            outputs = self.capsules(x)
            outputs = outputs.view(outputs.size(0),self.out_channels,self.num_capsules,outputs.size(2),outputs.size(3)).permute(0,2,3,4,1).contiguous()
            outputs = self.squash(outputs,dim=-1)

        return outputs


    def forward_old(self,x):
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