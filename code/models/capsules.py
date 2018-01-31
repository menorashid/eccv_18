import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import math

class Primary_Caps(nn.Module):
    
    def __init__(self,A, B, pose_mat_size = 4):
        super(Primary_Caps, self).__init__()
        self.B = B
        self.pose_mat_size = pose_mat_size
        self.pose = nn.Conv2d(in_channels=A,out_channels=self.pose_mat_size*self.pose_mat_size*self.B,kernel_size=1,stride=1)
        self.activation = []
        self.activation.append(nn.Conv2d(in_channels=A,out_channels=self.B,kernel_size=1,stride=1))
        self.activation.append(nn.Sigmoid())
        self.activation = nn.Sequential(*self.activation)

    def forward(self, x): #b,14,14,32
        pose = self.pose(x)
        activation = self.activation(x) #(b,1,12,12)*32
        activation = activation.view(activation.size(0),self.B,1,activation.size(2),activation.size(3))
        pose = pose.view(pose.size(0),self.B,self.pose_mat_size*self.pose_mat_size,pose.size(2),pose.size(3))
        output = torch.cat([pose,activation],dim=2)
        output = output.view(output.size(0),output.size(1)*output.size(2),output.size(3),output.size(4))
        
        
        return output



class Conv_Caps(nn.Module):
    def __init__(self, B, C, r, kernel, stride, pose_mat_size, class_it = False, coord_add = None):
        super(Conv_Caps, self).__init__()
        self.kernel = kernel
        self.pose_mat_size = pose_mat_size
        self.B = B
        self.C = C
        self.r = r
        self.epsilon = 1e-9
        self.ac_lambda = 0.01
        self.ac_lambda_step = 0.0025
        self.class_it = class_it
        
        in_channels = self.B * (self.pose_mat_size * self.pose_mat_size +1)
        self.kerneling_layer = nn.Conv2d(in_channels = in_channels, out_channels = kernel*kernel*in_channels, kernel_size = kernel, stride = stride, groups=in_channels)

        weight = torch.Tensor(in_channels,kernel*kernel,kernel,kernel).fill_(0)
        for i in range(kernel):
            for j in range(kernel):
                weight[:,i*kernel+j,i,j]=1
        
        weight = weight.view(in_channels*kernel*kernel,1,kernel,kernel)
        
        self.kerneling_layer.weight.data = weight    
        self.kerneling_layer.weight.requires_grad = False

        self.kerneling_layer.bias.data.fill_(0)
        self.kerneling_layer.bias.requires_grad = False
        self.w = nn.Parameter (torch.Tensor(1,self.kernel*self.kernel*self.B,self.C,self.pose_mat_size,self.pose_mat_size))


        self.beta_v = nn.Parameter(torch.Tensor(self.C,self.pose_mat_size*self.pose_mat_size))
        self.beta_a = nn.Parameter(torch.Tensor(self.C,))
        if coord_add is not None:
            self.coord_add = coord_add
            

    def kernel_it(self,x):
        kernel = self.kernel
        in_channels = self.B * (self.pose_mat_size * self.pose_mat_size +1)
        B = self.B
        
        output = self.kerneling_layer(x)
        output = output.view(output.size(0),in_channels,kernel*kernel,output.size(2),output.size(3))
        output = output.permute(0,2,1,3,4).contiguous()
        output = output.view(output.size(0),B*kernel*kernel,self.pose_mat_size*self.pose_mat_size+1,output.size(3),output.size(4))
        pose = output[:,:,:16,:,:]
        activation = output[:,:,16:,:,:]
        
        return pose,activation

    def mat_transform(self,pose):
        pose = pose.permute(0,3,4,1,2).contiguous()
        pose = pose.view(pose.size(0)*pose.size(1)*pose.size(2),pose.size(3),pose.size(4))
        pose = pose.view(pose.size(0),pose.size(1),1,self.pose_mat_size,self.pose_mat_size)
        pose = pose.repeat(1,1,self.C,1,1)
        w = self.w.expand(pose.size(0),self.w.size(1),self.w.size(2),self.w.size(3),self.w.size(4))
        
        votes = torch.matmul(pose,w)
        votes = votes.view(votes.size(0),votes.size(1),votes.size(2),votes.size(3)*votes.size(4))
        
        return votes


    def em_routing(self, votes, activation):
        
        # use_cuda = next(self.parameters()).is_cuda
        # print 'use_cuda',use_cuda
        activation = activation.permute(0,3,4,1,2).contiguous()
        activation = activation.view(activation.size(0)*activation.size(1)*activation.size(2),activation.size(3),activation.size(4))

        batch_size = votes.size(0)
        caps_num_i = activation.size(1)
        n_channels = votes.size(-1)
        C = self.C
        lambda_curr = self.ac_lambda


        for iter_num in range(self.r):
            # print iter_num
            # e_step

            if iter_num==0:
                # if use_cuda:
                #     r = torch.cuda.FloatTensor(batch_size,caps_num_i,C).fill_(1)/C
                # else:
                r = torch.Tensor(batch_size,caps_num_i,C).fill_(1)/float(C)
                r = torch.autograd.Variable(r)
            else:
                # print 'E step'
                # print 'miu.size(),sigma_square.size(),activation_out.size,votes.size()'
                # print miu.size(),sigma_square.size(),activation_out.size,votes.size()

                log_p_c_h = - torch.log ( torch.sqrt(sigma_square)) - \
                            (torch.pow(votes-miu,2)/(2*sigma_square))
                max_log_p_c_h = torch.max(torch.max(log_p_c_h,2)[0],2)[0]
                max_log_p_c_h = max_log_p_c_h.view(max_log_p_c_h.size(0),max_log_p_c_h.size(1),1,1)
                log_p_c_h = log_p_c_h - (max_log_p_c_h - math.log(10.0,10))
                
                p_c = torch.exp(torch.sum(log_p_c_h,3))

                ap = p_c * activation_out.view(batch_size,1,C)
                r = ap / (torch.sum(ap,2,keepdim=True)+self.epsilon)

            #m-step
            # print 'M step'
            # print 'r.size(),activation.size(),votes.size()'
            # print r.size(),activation.size(),votes.size()

            r = r*activation
            r = r/(torch.sum(r,2,keepdim=True)+self.epsilon)
            r_sum = torch.sum(r,1,keepdim=True)+self.epsilon
            r1 = r/r_sum
            r1 = r.view(r.size(0),r.size(1),r.size(2),1)

            miu = torch.sum(votes*r1,1,keepdim=True)
            sigma_square = torch.sum(torch.pow(votes-miu,2)*r1, 1,keepdim=True)+self.epsilon
            # print 'miu.size()',miu.size()
            # print 'sigma_square.size()',sigma_square.size()
            
            # if iter_num==(self.r-1):
            r_sum = r_sum.view(batch_size,C,1)
            cost_h = (self.beta_v + torch.log(torch.sqrt(sigma_square.view(batch_size,C,n_channels)))) * r_sum
            activation_out = nn.functional.softmax(self.ac_lambda * (self.beta_a - torch.sum(cost_h,2)),dim=-1)
            lambda_curr = lambda_curr+self.ac_lambda_step
            # else:
            #     activation_out = nn.functional.softmax(r_sum,dim=-1)

            # print 'r',torch.min(r).data,torch.max(r).data
            # raw_input()
            # print ' '

        return miu, activation_out

    def em_routing_maheen(self, votes, activation):
        debug = False
        # use_cuda = next(self.parameters()).is_cuda
        # print 'use_cuda',use_cuda
        activation = activation.permute(0,3,4,1,2).contiguous()
        activation = activation.view(activation.size(0)*activation.size(1)*activation.size(2),activation.size(3),activation.size(4))

        batch_size = votes.size(0)
        caps_num_i = activation.size(1)
        n_channels = votes.size(-1)
        C = self.C
        lambda_curr = self.ac_lambda


        for iter_num in range(self.r):
            # print iter_num
            # e_step

            if iter_num==0:
                # if use_cuda:
                #     r = torch.cuda.FloatTensor(batch_size,caps_num_i,C).fill_(1)/C
                # else:
                r = torch.Tensor(batch_size,caps_num_i,C).fill_(1)/float(C)
                r = torch.autograd.Variable(r)
            else:
                # print 'E step'
                # print 'miu.size(),sigma_square.size(),activation_out.size,votes.size()'
                # print miu.size(),sigma_square.size(),activation_out.size,votes.size()

                log_p_c_h = - torch.log ( torch.sqrt(sigma_square)) - \
                            (torch.pow(votes-miu,2)/(2*sigma_square))
                max_log_p_c_h = torch.max(torch.max(log_p_c_h,2)[0],2)[0]
                max_log_p_c_h = max_log_p_c_h.view(max_log_p_c_h.size(0),max_log_p_c_h.size(1),1,1)
                log_p_c_h = log_p_c_h - (max_log_p_c_h - math.log(10.0,10))
                
                p_c = torch.exp(torch.sum(log_p_c_h,3))

                ap = p_c * activation_out.view(batch_size,1,C)
                r = ap / (torch.sum(ap,2,keepdim=True)+self.epsilon)

            #m-step
            # print 'M step'
            # print 'r.size(),activation.size(),votes.size()'
            # print r.size(),activation.size(),votes.size()

            r = r*activation
            # r = r/(torch.sum(r,2,keepdim=True)+self.epsilon)
            r_sum = torch.sum(r,1,keepdim=True)+self.epsilon
            # r1 = r/r_sum
            # r1 = r.view(r.size(0),r.size(1),r.size(2),1)

            # print r.size()
            r_exp = r.view(r.size(0),r.size(1),r.size(2),1)
            miu = torch.sum(votes*r_exp,1,keepdim=True)
            deno = r_sum.view(r_sum.size(0),r_sum.size(1),r_sum.size(2),1).expand(r_sum.size(0),r_sum.size(1),r_sum.size(2),miu.size(-1))
            miu = miu/deno

            # print miu.size(),r.size(),votes.size(),r_sum.size(),deno.size()
            # raw_input()
            # /r_sum
            # print miu.expand(miu.size(0),votes.size(1),miu.size(2),miu.size(3)).size()
            # print r.size(),r_exp.size()
            sigma_square = torch.sum(torch.pow(votes-miu,2)*r_exp, 1, keepdim=True).clamp(min=1e-20)
             # +self.epsilon
            # /deno
            # +self.epsilon
            # print 'miu.size()',miu.size()
            # print 'sigma_square.size()',sigma_square.size()

            # if iter_num==(self.r-1):
            r_sum = r_sum.view(batch_size,C,1)
            internal = torch.log(torch.sqrt(sigma_square.view(batch_size,C,n_channels)))
            cost_h = (self.beta_v + internal) * r_sum
            if debug:
                print 'internal',torch.min(internal).data[0],torch.max(internal).data[0]
                print 'sigma_square',torch.min(sigma_square).data[0],torch.max(sigma_square).data[0]
                print 'cost_h',torch.min(cost_h).data[0],torch.max(cost_h).data[0]
                print 'self.beta_v',torch.min(self.beta_v).data[0],torch.max(self.beta_v).data[0]
                print 'r_sum',torch.min(r_sum).data[0],torch.max(r_sum).data[0]

            activation_out = nn.functional.sigmoid(lambda_curr * (self.beta_a - torch.sum(cost_h,2))).clamp(min=-1.,max=1.)
            # nn.functional.softmax(self.ac_lambda * (self.beta_a - torch.sum(cost_h,2)),dim=-1)
            # lambda_curr = lambda_curr+self.ac_lambda_step
            # else:
            #     activation_out = nn.functional.softmax(r_sum,dim=-1)
            if debug:
                # print 'r',torch.min(r).data,torch.max(r).data
                print 'activation_out',torch.min(activation_out).data[0],torch.max(activation_out).data[0]
                if torch.min(activation_out).data[0]==1.0 and torch.max(activation_out).data[0]==1.0:
                    raw_input()
                # raw_input()
                # print ' '

        return miu, activation_out

    def forward(self, x): #b,14,14,32
        # use_cuda = next(self.parameters()).is_cuda

        x_kerneled , activation = self.kernel_it(x)
        votes = self.mat_transform(x_kerneled)

        if self.class_it:
            votes = votes.view(x.size(0),x.size(2),x.size(3),votes.size(1),votes.size(2),self.pose_mat_size,self.pose_mat_size)
            coord_add = self.coord_add.view(1,self.coord_add.size(0),self.coord_add.size(1),1,1,self.coord_add.size(2))
            # if use_cuda:
            #     coord_add = coord_add.cuda()

            coord_add = coord_add.expand(votes.size(0),-1,-1,votes.size(3),votes.size(4),-1)

            votes[:,:,:,:,:,3,:2]= votes[:,:,:,:,:,3,:2] +coord_add
            votes = votes.view(votes.size(0)*votes.size(1)*votes.size(2),votes.size(3),votes.size(4),self.pose_mat_size*self.pose_mat_size)
        
        pose, activation_out = self.em_routing(votes, activation)
        pose = pose.view(activation.size(0),activation.size(3),activation.size(4),pose.size(2),pose.size(3))
        pose = pose.permute(0,3,4,1,2).contiguous()
        activation_out = activation_out.view(activation.size(0),activation.size(3),activation.size(4),activation_out.size(1),1)
        activation_out = activation_out.permute(0,3,4,1,2).contiguous()        

        output = torch.cat([pose,activation_out],dim=2)
        output = output.view(output.size(0),output.size(1)*output.size(2),output.size(3),output.size(4))

        if self.class_it:
            output = nn.functional.avg_pool2d(output, output.size(2))

        # print 'output.size()',output.size(),torch.min(output).data[0],torch.max(output).data[0]
        return output
        
