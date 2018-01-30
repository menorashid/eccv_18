import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import math
# from capsules import Primary_Caps,Conv_Caps


class PrimaryCaps_old(nn.Module):
    """
    Primary Capsule layer is nothing more than concatenate several convolutional
    layer together.
    Args:
        A:input channel
        B:number of types of capsules.
    
    """
    def __init__(self,A=32, B=32):
        super(PrimaryCaps, self).__init__()
        self.B = B
        self.capsules_pose = nn.ModuleList([nn.Conv2d(in_channels=A,out_channels=4*4,
                                                 kernel_size=1,stride=1) 
                                                 for i in range(self.B)])
        self.capsules_activation = nn.ModuleList([nn.Conv2d(in_channels=A,out_channels=1,
                                                 kernel_size=1,stride=1) for i 
                                                 in range(self.B)])

    def forward(self, x): #b,14,14,32
        poses = [self.capsules_pose[i](x) for i in range(self.B)]#(b,16,12,12) *32
        poses = torch.cat(poses, dim=1) #b,16*32,12,12
        activations = [self.capsules_activation[i](x) for i in range(self.B)] #(b,1,12,12)*32
        activations = nn.functional.sigmoid(torch.cat(activations, dim=1)) #b,32,12,12
        output = torch.cat([poses, activations], dim=1)
        return output

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
        self.ac_lambda_step = 0.01
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

        # if self.class_it:
        #     self.w = nn.Parameter (torch.Tensor(1,self.B,self.C,self.pose_mat_size,self.pose_mat_size))
        # else:
        self.w = nn.Parameter (torch.Tensor(1,self.kernel*self.kernel*self.B,self.C,self.pose_mat_size,self.pose_mat_size))


        self.beta_v = nn.Parameter(torch.Tensor(self.C,self.pose_mat_size*self.pose_mat_size))
        self.beta_a = nn.Parameter(torch.Tensor(self.C,))
        if coord_add is not None:
            self.coord_add = torch.autograd.Variable(coord_add)
            

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
        
        # if self.class_it:
        #     w = self.w.repeat(pose.size(0),self.kernel*self.kernel,1,1,1)
        # else:
        w = self.w.expand(pose.size(0),self.w.size(1),self.w.size(2),self.w.size(3),self.w.size(4))
        
        votes = torch.matmul(pose,w)
        votes = votes.view(votes.size(0),votes.size(1),votes.size(2),votes.size(3)*votes.size(4))
        
        return votes


    def em_routing(self, votes, activation):
        
        # use_cuda = next(self.parameters()).is_cuda

        activation = activation.permute(0,3,4,1,2).contiguous()
        activation = activation.view(activation.size(0)*activation.size(1)*activation.size(2),activation.size(3),activation.size(4))

        batch_size = votes.size(0)
        caps_num_i = activation.size(1)
        n_channels = votes.size(-1)
        C = self.C

        for iter_num in range(self.r):
            # e_step
            if iter_num==0:
                r = torch.autograd.Variable(torch.Tensor(batch_size,caps_num_i,C).fill_(1)/C)
                # if use_cuda:
                #     r = r.cuda()
            else:
                log_p_c_h = - torch.log ( torch.sqrt(sigma_square)) - \
                            (torch.pow(votes-miu,2)/(2*sigma_square))
                # print 'log_p_c_h.size()',log_p_c_h.size()
                max_log_p_c_h = torch.max(torch.max(log_p_c_h,2)[0],2)[0]
                max_log_p_c_h = max_log_p_c_h.view(max_log_p_c_h.size(0),max_log_p_c_h.size(1),1,1)
                log_p_c_h = log_p_c_h - (max_log_p_c_h - math.log(10.0,10))
                
                p_c = torch.exp(torch.sum(log_p_c_h,3))

                ap = p_c * activation_out.view(batch_size,1,C)
                r = ap / (torch.sum(ap,2,keepdim=True)+self.epsilon)

            #m-step
            r = r*activation
            r = r/(torch.sum(r,2,keepdim=True)+self.epsilon)
            r_sum = torch.sum(r,1,keepdim=True)+self.epsilon
            r1 = r/r_sum
            r1 = r.view(r.size(0),r.size(1),r.size(2),1)

            miu = torch.sum(votes*r1,1,keepdim=True)
            sigma_square = torch.sum(torch.pow(votes-miu,2)*r1, 1,keepdim=True)+self.epsilon
            
            if iter_num==(self.r-1):
                r_sum = r_sum.view(batch_size,C,1)
                cost_h = (self.beta_v + torch.log(torch.sqrt(sigma_square.view(batch_size,C,n_channels)))) * r_sum
                activation_out = nn.functional.softmax(self.ac_lambda * (self.beta_a - torch.sum(cost_h,2)),dim=-1)
            else:
                activation_out = nn.functional.softmax(r_sum,dim=-1)

        return miu, activation_out


    def forward(self, x): #b,14,14,32
        
        # if self.class_it:
        #     temp = self.pose_mat_size*self.pose_mat_size+1
        #     assert x.size(1)%temp==0
        #     x = x.view(x.size(0),x.size(1)/temp,temp,x.size(2),x.size(3))
        #     x_kerneled = x[:,:,:self.pose_mat_size*self.pose_mat_size,:,:]
        #     activation = x[:,:,self.pose_mat_size*self.pose_mat_size:,:,:]
        # else:
        x_kerneled , activation = self.kernel_it(x)
        print 'x_kerneled.size()',x_kerneled.size()
        votes = self.mat_transform(x_kerneled)

        # votes = votes
        
        if self.class_it:
            print 'votes.size()',votes.size(),x.size()
            votes = votes.view(x.size(0),x.size(2),x.size(3),votes.size(1),votes.size(2),self.pose_mat_size,self.pose_mat_size)
            print 'votes.size()',votes.size()
            coord_add = self.coord_add.view(1,self.coord_add.size(0),self.coord_add.size(1),1,1,self.coord_add.size(2))
            print coord_add.size()
            coord_add = coord_add.expand(votes.size(0),-1,-1,votes.size(3),votes.size(4),-1)
            votes[:,:,:,:,:,3,:2]= votes[:,:,:,:,:,3,:2] +coord_add
            votes = votes.view(votes.size(0)*votes.size(1)*votes.size(2),votes.size(3),votes.size(4),self.pose_mat_size*self.pose_mat_size)
            print 'votes.size()',votes.size()
            print coord_add.size()
        print 'activation.size()',activation.size()

        pose, activation_out = self.em_routing(votes, activation)
        pose = pose.view(activation.size(0),activation.size(3),activation.size(4),pose.size(2),pose.size(3))
        pose = pose.permute(0,3,4,1,2).contiguous()
        activation_out = activation_out.view(activation.size(0),activation.size(3),activation.size(4),activation_out.size(1),1)
        activation_out = activation_out.permute(0,3,4,1,2).contiguous()        

        output = torch.cat([pose,activation_out],dim=2)
        output = output.view(output.size(0),output.size(1)*output.size(2),output.size(3),output.size(4))

        if self.class_it:
            output = nn.functional.avg_pool2d(output, output.size(2))

        if debug:
            print x.size()
            print 'pose.size()',pose.size()
            print 'activation.size()',activation.size()

            print 'votes.size()',votes.size()
        #     # print 'miu.size()',miu.size()
            print 'activation_out.size()',activation_out.size()
            print 'output.size()',output.size()
        

        return output
        # ,pose,activation



class Matrix_Capsule_Model(nn.Module):

    def __init__(self,n_classes,out_channels,r,pose_mat_size=4):
        super(Matrix_Capsule_Model, self).__init__()
        out_size = 3

        coord_add = torch.Tensor(np.array([[[8., 8.], [12., 8.], [16., 8.]],
                          [[8., 12.], [12., 12.], [16., 12.]],
                          [[8., 16.], [12., 16.], [16., 16.]]]))



        self.pose_mat_size = pose_mat_size
        self.features = []
        self.features.append(nn.Conv2d(in_channels=1, out_channels=out_channels[0],
                               kernel_size=5, stride=2))
        self.features.append(nn.ReLU(True))
        self.features = nn.Sequential(*self.features)

        self.primary_caps = Primary_Caps(out_channels[0], out_channels[1])
        self.conv_caps1 = Conv_Caps(out_channels[1], out_channels[2], r, 3, 2, self.pose_mat_size)
        self.conv_caps2 = Conv_Caps(out_channels[2], out_channels[3], r, 3, 1, self.pose_mat_size)
        self.class_caps = Conv_Caps(out_channels[3], n_classes, r, 1, 1, self.pose_mat_size, class_it=True, coord_add=coord_add)

        
    def forward(self, x):
        x = self.features(x)
        print x.size()
        x = self.primary_caps(x)
        print x.size()
        print 'CONV CAPS1'
        x = self.conv_caps1(x)
        print x.size()
        print 'CONV CAPS2'
        x = self.conv_caps2(x)
        print x.size()
        print 'CLASS CAPS1'
        x = self.class_caps(x)
        print x.size()

        temp = self.pose_mat_size*self.pose_mat_size+1
        output = x.view(x.size(0),x.size(1)/temp,temp)
        activation = output[:,:,-1]
        print activation.size()

        return activation

    def spread_loss(self,x,target,m):
        b = x.size(0)
        target_t = target.type(torch.LongTensor)
        target_t = target_t.cuda()
        rows = torch.LongTensor(np.array(range(b))).cuda()
        a_t = x[rows,target_t]
        
        # a_t = torch.cat([x[i][target[i]] for i in range(b)]) #b
        a_t_stack = a_t.view(b,1).expand(b,x.size(1)).contiguous() #b,10
        u = m-(a_t_stack-x) #b,10
        u = nn.functional.relu(u)**2
        u[rows,target_t]=0
        loss = torch.sum(u)/b
        # print u
        # mask = u.ge(0).float() #max(u,0) #b,10
        # loss = ((mask*u)**2)/b - m**2  #float
        return loss

class Network:
    def __init__(self,n_classes=8,out_channels=[32,8,16,16],r=3):
        model = Matrix_Capsule_Model(n_classes,out_channels,r)

        
        for idx_m,m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                nn.init.constant(m.bias.data,0.)
            elif isinstance(m, nn.BatchNorm2d):
                
                nn.init.constant(m.weight.data,1.)
                nn.init.constant(m.bias.data,0.)
                
        self.model = model
        
    
    # def get_lr_list(self, lr):
    #     lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]\
    #             +[{'params': self.model.classifier.parameters(), 'lr': lr[1]}]
    #     return lr_list



    


def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    options = {'mnist': ([[[8., 8.], [12., 8.], [16., 8.]],
                          [[8., 12.], [12., 12.], [16., 12.]],
                          [[8., 16.], [12., 16.], [16., 16.]]], 28.),
               'smallNORB': ([[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
                              [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
                              [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
                              [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]], 32.)
               }
    mnist_coord_add = np.array(options['mnist'][0])
    print mnist_coord_add.shape

    smallNORB_coord_add = np.array(options['smallNORB'][0])
    print smallNORB_coord_add.shape
    
    # return
    # B = 3
    # caps_layer = Conv_Caps(B = B, C = 32, r = 3, kernel = 3, stride = 2)
    # # print caps_layer.kerneling_layer.weight.size()
    
    
    # in_channels=B*(4*4+1)
    # kernel = 3
    # # # weight = torch.Tensor(kernel*kernel,in_channels,kernel,kernel).fill_(0)
    # # weight = torch.Tensor(in_channels,kernel*kernel,kernel,kernel).fill_(0)
    # # # weight = np.zeros(shape=(in_channels,kernel*kernel,kernel,kernel))
    # # print weight.size()
    # # for i in range(kernel):
    # #     for j in range(kernel):
    # #         # weight[i*kernel+j,:,i,j]=1
    # #         weight[:,i*kernel+j,i,j]=1
    
    # # # for i in range(weight.shape[1]):
    # # #     print i
    # # #     print weight[4,i]

    # # weight = weight.view(in_channels*kernel*kernel,1,kernel,kernel)
    # # print weight[:10,0,:,:]
    # # raw_input()
    # # print weight.size()
    
    # # caps_layer.kerneling_layer.weight.data = weight    
    # # caps_layer.kerneling_layer.bias.data.fill_(0)

    # batch_size = 3
    # input = np.zeros((batch_size,in_channels,5,5))
    # for bs in range(batch_size):
    #     for filter_num in range(in_channels):
    #         for row in range(input.shape[2]):
    #             for col in range(input.shape[3]):
    #                 input[bs,filter_num,row,col]= bs*1000 + filter_num+1+0.1*(row+1)+0.01*(col+1)
    #         # bs+1

    
    # # print 'input[0,0]'
    # # print input[0,0]    
    # # print 'input[0,1]'
    # # print input[0,1]    
    # # print 'input[0,2]'
    # # print input[0,2]    
    # # print 'input[1,0]'
    # # print input[1,0]    
    # # print 'input[1,1]'
    # # print input[1,1]
    # # print 'input[1,2]'
    # # print input[1,2]
    # # raw_input()
            

    # input = Variable(torch.Tensor(input))
    # # output,
    # output = caps_layer(input)
    # print output.size()
    # # print activation.size()
    # # print pose.size()


    # return
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(2)

    net = Network()
    net.model = net.model.cuda()
    print net.model
    input = np.zeros((10,1,28,28))
    labels = np.array(range(8)+[0,1],dtype=np.int)
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    labels = Variable(torch.Tensor(labels).cuda())
    print labels.size()
    output = net.model(input)
    loss = net.model.spread_loss(output,labels,0.2)
    print loss

    loss.backward()


    # ,activation, pose = 
    # output = output.cpu()
    # activation = activation.cpu()
    # pose = pose.cpu()
    print output.size()
    # ,activation.data.shape, pose.data.shape

    # B = 8
    # pose_mat_size = 16
    # assert np.all(output.data.numpy()[:,:pose_mat_size*B,:,:]==pose.data.numpy())

    # activation_rs = activation.view(activation.size(0),B,1,activation.size(2),activation.size(3))
    # print activation_rs.size()
    # pose_rs = pose.view(pose.size(0),B,pose_mat_size,pose.size(2),pose.size(3))
    # print pose_rs.size()
    # output_new = torch.cat([pose_rs,activation_rs],dim=2)
    # print output_new.size()
    # output_new = output_new.view(output_new.size(0),output_new.size(1)*output_new.size(2),output_new.size(3),output_new.size(4))
    # print output_new.size()
    # assert np.all(output.data.numpy()[:,:16,:,:]==pose.data.numpy()[:,0,:,:,:])    
    # assert np.all(output.data.numpy()[:,16,:,:]==activation.data.numpy()[:,0,0,:,:])    
    # assert np.all(output.data.numpy()[:,5*17:5*17+16,:,:]==pose.data.numpy()[:,5,:,:,:])    
    # assert np.all(output.data.numpy()[:,5*17+16,:,:]==activation.data.numpy()[:,5,0,:,:])    



    

if __name__=='__main__':
    main()