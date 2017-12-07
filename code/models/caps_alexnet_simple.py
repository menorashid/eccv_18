from torchvision import models
import torch.nn as nn
from capsule_layer import CapsuleLayer

import torch
import torch.nn as nn
from torch.autograd import Variable

from conv_layer import ConvLayer
from capsule_layer import CapsuleLayer
import utils

import numpy as np

class Net(nn.Module):
    """
    A simple CapsNet with 3 layers
    """

    def __init__(self, num_conv_in_channel, num_conv_out_channel, num_primary_unit, primary_unit_size,
                 num_classes, output_unit_size, num_routing,
                 use_reconstruction_loss, regularization_scale, cuda_enabled,caps_channels = 32, kernel_size = 9, stride = 2):
        """
        In the constructor we instantiate one ConvLayer module and two CapsuleLayer modules
        and assign them as member variables.
        """
        super(Net, self).__init__()

        self.cuda_enabled = cuda_enabled

        # Configurations used for image reconstruction.
        self.use_reconstruction_loss = use_reconstruction_loss
        self.image_width = 224 # MNIST digit image width
        self.image_height = 224 # MNIST digit image height
        self.image_channel = 3 # MNIST digit image channel
        self.regularization_scale = regularization_scale

        # Layer 1: Conventional Conv2d layer
        # self.conv1 = ConvLayer(in_channel=num_conv_in_channel,
        #                        out_channel=num_conv_out_channel,
        #                        kernel_size=9)


        features = list(models.alexnet(pretrained=True).features)[:-3]
        # features.pop()
        # features.pop()
        # features.pop()
        self.conv1 = nn.Sequential(*features)
        


        # ConvLayer(in_channel=num_conv_in_channel,
        #                        out_channel=num_conv_out_channel,
        #                        kernel_size=9)

        # PrimaryCaps
        # Layer 2: Conv2D layer with `squash` activation
        # print ('n_channel=',num_conv_out_channel)
        # print ('num_unit=',num_primary_unit)
        # print ('unit_size=',primary_unit_size)
        # print ('use_routing=',False)
        # print ('num_routing=',num_routing)
        # print ('cuda_enabled=',cuda_enabled)
        self.primary = CapsuleLayer(in_unit=0,
                                    in_channel=num_conv_out_channel,
                                    num_unit=num_primary_unit,
                                    unit_size=primary_unit_size, # capsule outputs
                                    use_routing=False,
                                    num_routing=num_routing,
                                    cuda_enabled=cuda_enabled,caps_channels = caps_channels,kernel_size=kernel_size, stride = stride)

        # DigitCaps
        # Final layer: Capsule layer where the routing algorithm is.
        # print ('SECOND')
        # print ('in_unit=',num_primary_unit)
        # print ('in_channel=',primary_unit_size)
        # print ('num_unit=',num_classes)
        # print ('unit_size=',output_unit_size)
        # print ('use_routing=',True)
        # print ('num_routing=',num_routing)
        # print ('cuda_enabled=',cuda_enabled)
        

        self.digits = CapsuleLayer(in_unit=num_primary_unit,
                                   in_channel=primary_unit_size,
                                   num_unit=num_classes,
                                   unit_size=output_unit_size, # 16D capsule per digit class
                                   use_routing=True,
                                   num_routing=num_routing,
                                   cuda_enabled=cuda_enabled)

        # Reconstruction network
        if use_reconstruction_loss:
            # The output of the digit capsule is fed into a decoder consisting of
            # 3 fully connected layers that model the pixel intensities.
            fc1_output_size = 512
            fc2_output_size = 1024
            self.fc1 = nn.Linear(num_classes * output_unit_size, fc1_output_size)
            self.fc2 = nn.Linear(fc1_output_size, fc2_output_size)
            self.fc3 = nn.Linear(fc2_output_size, 784)
            # Activation functions
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the computation performed at every forward pass.
        """
        out_conv1 = self.conv1(x)
        out_primary_caps = self.primary(out_conv1)
        out_digit_caps = self.digits(out_primary_caps)
        return out_digit_caps

    def loss(self, images, out_digit_caps, target, size_average=True):
        """Custom loss function"""
        # out_digit_caps is the output from DigitCaps layer during the forward pass.
        # out_digit_caps is the input to the loss function.
        m_loss = self.margin_loss(out_digit_caps, target, size_average)
        total_loss = m_loss

        if self.use_reconstruction_loss:
            # In order to keep in line with the paper,
            # they scale down the reconstruction loss by 0.0005
            # so that it does not dominate the margin loss.
            recon_loss = self.reconstruction_loss(images, out_digit_caps, size_average)
            total_loss = m_loss + recon_loss * self.regularization_scale

        return total_loss

    def margin_loss(self, input, target, size_average=True):
        """
        Class loss

        Implement section 3 'Margin loss for digit existence' in the paper.
        """
        batch_size = input.size(0)

        # Implement equation 4 in the paper.

        # ||vc||
        v_c = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms.
        zero = Variable(torch.zeros(1))
        if self.cuda_enabled:
            zero = zero.cuda()
        m_plus = 0.9
        m_minus = 0.1
        loss_lambda = 0.5
        max_left = torch.max(m_plus - v_c, zero).view(batch_size, -1)**2
        max_right = torch.max(v_c - m_minus, zero).view(batch_size, -1)**2
        t_c = target
        # Lc is margin loss for each digit of class c
        l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
        l_c = l_c.sum(dim=1)

        if size_average:
            l_c = l_c.mean()

        return l_c

    def reconstruction_loss(self, images, input, size_average=True):
        """
        Implement section 4.1 'Reconstruction as a regularization method' in the paper.
        Implement Decoder structure in Figure 2 to reconstruct a digit from
        the DigitCaps layer representation.

        Based on naturomics's implementation.
        """

        """
        First, do masking.
        """
        # Method 1: mask with y.
        # Note: we have not implement method 2 which is masking with true label.
        masked_caps = utils.mask(input, self.cuda_enabled)

        """
        Second, reconstruct the images with 3 Fully Connected layers.
        """
        vector_j = masked_caps.view(input.size(0), -1) # reshape the masked_caps tensor
        fc1_out = self.relu(self.fc1(vector_j))
        fc2_out = self.relu(self.fc2(fc1_out))
        decoded = self.sigmoid(self.fc3(fc2_out))

        recon_img = decoded.view(-1, self.image_channel, self.image_height, self.image_width)

        """
        Save reconstructed images.
        """
        utils.save_image(recon_img, 'results/reconstructed_images.png')

        """
        Calculate reconstruction loss.
        """
        # Minimize the sum of squared differences between the
        # reconstructed image (outputs of the logistic units) and the input image (origin).
        error = (recon_img - images).view(recon_img.size(0), -1)
        squared = error**2

        recon_error = torch.sum(squared, dim=1)

        # Mean squared error
        if size_average:
            recon_error = recon_error.mean()

        return recon_error



class Network:
    def __init__(self):
        # model_ft = models.alexnet(pretrained=True)
        
        # new_features = list(model_ft.features.children())
        # new_features.pop()

        # new_classifier = list(model_ft.classifier.children())
        # new_classifier.pop()
        # new_layer = nn.Linear(4096,2)
        # nn.init.xavier_normal(list(new_layer.parameters())[0])
        # nn.init.constant(list(new_layer.parameters())[1],0.)
        # new_classifier.append(new_layer)
        # model_ft.classifier = nn.Sequential(*new_classifier)
        num_conv_in_channel = 1
        num_conv_out_channel = 256
        num_primary_unit = 8
        primary_unit_size = 1152
        # 9216
        num_classes = 2
        output_unit_size = 32
        num_routing = 3
        use_reconstruction_loss = False
        regularization_scale = 0.0005
        cuda_enabled = True
        kernel_size = 3


        self.model = Net(num_conv_in_channel, num_conv_out_channel, num_primary_unit, primary_unit_size,
                 num_classes, output_unit_size, num_routing,
                 use_reconstruction_loss, regularization_scale, cuda_enabled,kernel_size = kernel_size)

        alexnet = models.alexnet(pretrained=True)
        layer_curr = list(alexnet.features)[-3]
        weight,bias = [param.data.numpy() for param in layer_curr.parameters()]
        weights_biases = [np.split(weight, 8, axis=0),np.split(bias, 8, axis=0)]
        for idx_model_curr,module_curr in enumerate(list(self.model.primary.conv_units)):
            for idx_param,param in enumerate(module_curr.parameters()):
                param.data = torch.FloatTensor(weights_biases[idx_param][idx_model_curr])
        



    def get_lr_list(self, lr):
        lr_list= [{'params': self.model.conv1.parameters(), 'lr': lr[0]}]\
        +[{'params': self.model.primary.parameters(), 'lr': lr[1]}]\
        +[{'params': self.model.digits.parameters(), 'lr': lr[2]}]
        return lr_list

def main():
    network = Network()
    # alexnet = models.alexnet(pretrained=True)
    # layer_curr = list(alexnet.features)[-3]

    # weight,bias = [param.data.numpy() for param in layer_curr.parameters()]
    # # print weight.shape,bias.shape 
    # weights_biases = [np.split(weight, 8, axis=0),np.split(bias, 8, axis=0)]
    
        
    # # print biases

    # # #split weight = weight 

    # # # for param in layer_curr.parameters():
    # # #     print param.data.shape

    # for idx_model_curr,module_curr in enumerate(list(network.model.primary.conv_units)):
    #     for idx_param,param in enumerate(module_curr.parameters()):
    #         param.data = torch.FloatTensor(weights_biases[idx_param][idx_model_curr])
            # print torch.min(list(list(network.model.primary.conv_units)[-1].parameters())[-1]),torch.max(list(list(network.model.primary.conv_units)[-1].parameters())[-1])

    # # network.model
    # print torch.min(list(list(network.model.primary.conv_units)[-1].parameters())[-1]),torch.max(list(list(network.model.primary.conv_units)[-1].parameters())[-1])
    network.model = network.model.cuda()
    print network.model
    lr_list = network.get_lr_list([0,0.001,0.001])
    print lr_list

    data = Variable(torch.Tensor(np.zeros((10,3,224,224)))).cuda()
    output = network.model(data)
    print output.data.shape
    # output = network.model.primary(output)
    # # data = Variable(torch.Tensor(np.zeros((10,1,28,28)))).cuda()
    # # output = network.model(data)
    # print output.data.shape
    


if __name__=='__main__':
    main()