import torch.nn as nn

import numpy as np
import scipy.misc
import torch
debug = True
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
        if debug:
            print 'pose.size()',pose.size()
        activation = self.activation(x) #(b,1,12,12)*32
        if debug:
            print 'activation.size()',activation.size()

        activation = activation.view(activation.size(0),self.B,1,activation.size(2),activation.size(3))
        if debug:
            print 'activation.size()',activation.size()

        pose = pose.view(pose.size(0),self.B,self.pose_mat_size*self.pose_mat_size,pose.size(2),pose.size(3))
        if debug:
            print 'pose.size()',pose.size()
        
        output = torch.cat([pose,activation],dim=2)
        output = output.view(output.size(0),output.size(1)*output.size(2),output.size(3),output.size(4))
        if debug:
            print 'output.size()',output.size()
        
        if debug:
            return output, activation, pose
        else:
            return output


class Conv_Caps(nn.Module):
    def __init__(self, B, C, r, kernel=3, stride=2, pose_mat_size = 4):
        super(Conv_Caps, self).__init__()
        self.kernel = kernel
        self.pose_mat_size = pose_mat_size
        self.B = B
        self.C = C
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
        self.w = torch.autograd.Variable (torch.Tensor(1,self.kernel*self.kernel*self.B,self.C,self.pose_mat_size,self.pose_mat_size))
        # print self.w.size()



    def kernel_it(self,x):
        kernel = self.kernel
        in_channels = self.B * (self.pose_mat_size * self.pose_mat_size +1)
        B = self.B
        
        output = self.kerneling_layer(x)
        output = output.view(output.size(0),in_channels,kernel*kernel,output.size(2),output.size(3))
        if debug:
            print 'output.size()',output.size()
        output = output.permute(0,2,1,3,4).contiguous()
        if debug:
            print 'output.size()',output.size()

        output = output.view(output.size(0),B*kernel*kernel,17,output.size(3),output.size(4))
        if debug:
            print 'output.size()',output.size()
        
        
        pose = output[:,:,:16,:,:]
        activation = output[:,:,16:,:,:]
        if debug:
            print 'pose.size()',pose.size()
            print 'activation.size()',activation.size()
        
        return pose,activation

    def mat_transform(self,pose):
        pose = pose.permute(0,3,4,1,2).contiguous()
        print pose.size()
        pose = pose.view(pose.size(0)*pose.size(1)*pose.size(2),pose.size(3),pose.size(4))
        print pose.size()
        pose = pose.view(pose.size(0),pose.size(1),1,self.pose_mat_size,self.pose_mat_size)
        print pose.size()
        return pose

    def forward(self, x): #b,14,14,32
        
        pose , activation = self.kernel_it(x)
        votes = self.mat_transform(pose)
        if debug:
            print 'votes.size()',votes.size()
            
        return pose, activation



class Matrix_Capsule_Model(nn.Module):

    def __init__(self,n_classes,out_channels,r):
        super(Matrix_Capsule_Model, self).__init__()

        self.conv1 = []
        self.conv1.append(nn.Conv2d(in_channels=1, out_channels=out_channels[0],
                               kernel_size=5, stride=2))
        self.conv1.append(nn.ReLU(True))
        self.conv1 = nn.Sequential(*self.conv1)

        self.primary_caps = Primary_Caps(out_channels[0],out_channels[1])
        
        # self.conv2 = nn.Conv2d(in_channels=A, out_channels=A,
        #                        kernel_size=5, stride=2)
        

        # self.features = []
        # self.features.append(nn.Conv2d(1, 64, 5, padding = 2))
        # self.features.append(nn.ReLU(True))
        # if bn:
        #     self.features.append(nn.BatchNorm2d(64,affine=True,momentum=0.1))
        # self.features.append(nn.MaxPool2d(2,2))
        
        # self.features.append(nn.Conv2d(64, 128, 5, padding = 2))
        # self.features.append(nn.ReLU(True))
        # if bn:
        #     self.features.append(nn.BatchNorm2d(128,affine=True,momentum=0.1))
        # self.features.append(nn.MaxPool2d(2,2))
        
        # self.features.append(nn.Conv2d(128, 256, 5, padding = 2))
        # self.features.append(nn.ReLU(True))
        # if bn:
        #     self.features.append(nn.BatchNorm2d(256,affine=True,momentum=0.1))
        # self.features.append(nn.AvgPool2d(12,12)) # quadrant pooling
        
        # self.features = nn.Sequential(*self.features)
        # self.classifier = []
        # self.classifier.append(nn.Linear(1024,300))
        # self.classifier.append(nn.ReLU(True))
        # self.classifier.append(nn.Dropout(0.5))
        # if bn:
        #     self.classifier.append(nn.BatchNorm1d(300,affine=True,momentum=0.1))
        # self.classifier.append(nn.Linear(300,n_classes))
        
        # self.classifier = nn.Sequential(*self.classifier)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.primary_caps(x)
        # x = x.view(x.size(0), 1024)
        # x = self.classifier(x)
        return x

class Network:
    def __init__(self,n_classes=8,out_channels=[64,8,16,16],r=3):
        model = Matrix_Capsule_Model(n_classes,out_channels,r)

        
        for idx_m,m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                # print m,1

                # print m.weight.data.shape, torch.min(m.weight.data), torch.max(m.weight.data)
                # print m.bias.data.shape, torch.min(m.bias.data), torch.max(m.bias.data)

                nn.init.xavier_normal(m.weight.data)
                nn.init.constant(m.bias.data,0.)

                # print m.weight.data.shape, torch.min(m.weight.data), torch.max(m.weight.data)
                # print m.bias.data.shape, torch.min(m.bias.data), torch.max(m.bias.data)

            elif isinstance(m, nn.BatchNorm2d):
                # print m,2
                nn.init.constant(m.weight.data,1.)
                nn.init.constant(m.bias.data,0.)
            # print 'break'
                
        self.model = model
        
    
    # def get_lr_list(self, lr):
    #     lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]\
    #             +[{'params': self.model.classifier.parameters(), 'lr': lr[1]}]
    #     return lr_list




def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    B = 3
    caps_layer = Conv_Caps(B = B, C = 16, r = 3, kernel = 3, stride = 2)
    print caps_layer.kerneling_layer.weight.size()
    
    
    in_channels=B*(4*4+1)
    kernel = 3
    # # weight = torch.Tensor(kernel*kernel,in_channels,kernel,kernel).fill_(0)
    # weight = torch.Tensor(in_channels,kernel*kernel,kernel,kernel).fill_(0)
    # # weight = np.zeros(shape=(in_channels,kernel*kernel,kernel,kernel))
    # print weight.size()
    # for i in range(kernel):
    #     for j in range(kernel):
    #         # weight[i*kernel+j,:,i,j]=1
    #         weight[:,i*kernel+j,i,j]=1
    
    # # for i in range(weight.shape[1]):
    # #     print i
    # #     print weight[4,i]

    # weight = weight.view(in_channels*kernel*kernel,1,kernel,kernel)
    # print weight[:10,0,:,:]
    # raw_input()
    # print weight.size()
    
    # caps_layer.kerneling_layer.weight.data = weight    
    # caps_layer.kerneling_layer.bias.data.fill_(0)

    batch_size = 3
    input = np.zeros((batch_size,in_channels,5,5))
    for bs in range(batch_size):
        for filter_num in range(in_channels):
            for row in range(input.shape[2]):
                for col in range(input.shape[3]):
                    input[bs,filter_num,row,col]= bs*1000 + filter_num+1+0.1*(row+1)+0.01*(col+1)
            # bs+1

    
    # print 'input[0,0]'
    # print input[0,0]    
    # print 'input[0,1]'
    # print input[0,1]    
    # print 'input[0,2]'
    # print input[0,2]    
    # print 'input[1,0]'
    # print input[1,0]    
    # print 'input[1,1]'
    # print input[1,1]
    # print 'input[1,2]'
    # print input[1,2]
    # raw_input()
            

    input = Variable(torch.Tensor(input))
    # output,
    pose, activation = caps_layer(input)
    # print output.size()
    # print activation.size()
    # print pose.size()


    return
    
    net = Network()
    net.model = net.model.cuda()
    print net.model
    input = np.zeros((10,1,48,48))
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    output,activation, pose = net.model(input)
    output = output.cpu()
    activation = activation.cpu()
    pose = pose.cpu()
    print output.data.shape,activation.data.shape, pose.data.shape

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
    assert np.all(output.data.numpy()[:,:16,:,:]==pose.data.numpy()[:,0,:,:,:])    
    assert np.all(output.data.numpy()[:,16,:,:]==activation.data.numpy()[:,0,0,:,:])    
    assert np.all(output.data.numpy()[:,5*17:5*17+16,:,:]==pose.data.numpy()[:,5,:,:,:])    
    assert np.all(output.data.numpy()[:,5*17+16,:,:]==activation.data.numpy()[:,5,0,:,:])    



    

if __name__=='__main__':
    main()