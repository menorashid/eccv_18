
import torch
from torch.utils.serialization import load_lua
from torchvision import models
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
# import torch.nn.functional as F

# class AvgPool3d(nn.Module):
#     r"""Applies a 3D average pooling over an input signal composed of several input
#     planes.
#     In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
#     output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
#     can be precisely described as:
#     .. math::
#         \begin{array}{ll}
#         out(N_i, C_j, d, h, w)  = 1 / (kD * kH * kW) * \sum_{{k}=0}^{kD-1} \sum_{{m}=0}^{kH-1} \sum_{{n}=0}^{kW-1}
#                                input(N_i, C_j, stride[0] * d + k, stride[1] * h + m, stride[2] * w + n)
#         \end{array}
#     | If :attr:`padding` is non-zero, then the input is implicitly zero-padded on all three sides
#       for :attr:`padding` number of points
#     The parameters :attr:`kernel_size`, :attr:`stride` can either be:
#         - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
#         - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
#           the second `int` for the height dimension and the third `int` for the width dimension
#     Args:
#         kernel_size: the size of the window
#         stride: the stride of the window. Default value is :attr:`kernel_size`
#         padding: implicit zero padding to be added on all three sides
#         ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
#         count_include_pad: when True, will include the zero-padding in the averaging calculation
#     Shape:
#         - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
#         - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
#           :math:`D_{out} = floor((D_{in} + 2 * padding[0] - kernel\_size[0]) / stride[0] + 1)`
#           :math:`H_{out} = floor((H_{in} + 2 * padding[1] - kernel\_size[1]) / stride[1] + 1)`
#           :math:`W_{out} = floor((W_{in} + 2 * padding[2] - kernel\_size[2]) / stride[2] + 1)`
#     Examples::
#         >>> # pool of square window of size=3, stride=2
#         >>> m = nn.AvgPool3d(3, stride=2)
#         >>> # pool of non-square window
#         >>> m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))
#         >>> input = autograd.Variable(torch.randn(20, 16, 50,44, 31))
#         >>> output = m(input)
#     """

#     def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
#                  count_include_pad=True):
#         super(AvgPool3d, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride or kernel_size
#         self.padding = padding
#         self.ceil_mode = ceil_mode
#         self.count_include_pad = count_include_pad

#     def forward(self, input):
#         return F.avg_pool3d(input, self.kernel_size, self.stride,
#                             self.padding, self.ceil_mode, self.count_include_pad)

#     def __setstate__(self, d):
#         super(AvgPool3d, self).__setstate__(d)
#         self.__dict__.setdefault('padding', 0)
#         self.__dict__.setdefault('ceil_mode', False)
#         self.__dict__.setdefault('count_include_pad', True)

#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#             + 'kernel_size=' + str(self.kernel_size) \
#             + ', stride=' + str(self.stride) \
#             + ', padding=' + str(self.padding) \
#             + ', ceil_mode=' + str(self.ceil_mode) \
#             + ', count_include_pad=' + str(self.count_include_pad) + ')'


# class LRN(nn.Module):
#     def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
#         super(LRN, self).__init__()
#         self.ACROSS_CHANNELS = ACROSS_CHANNELS
#         if ACROSS_CHANNELS:
#             self.average=AvgPool3d(kernel_size=(local_size, 1, 1),
#                     stride=1,
#                     padding=(int((local_size-1.0)/2), 0, 0))
#         else:
#             self.average=nn.AvgPool2d(kernel_size=local_size,
#                     stride=1,
#                     padding=int((local_size-1.0)/2))
#         self.alpha = alpha
#         self.beta = beta


#     def forward(self, x):
#         if self.ACROSS_CHANNELS:
#             div = x.pow(2).unsqueeze(1)
#             div = self.average(div).squeeze(1)
#             div = div.mul(self.alpha).add(1.0).pow(self.beta)
#         else:
#             div = x.pow(2)
#             div = self.average(div)
#             div = div.mul(self.alpha).add(1.0).pow(self.beta)
#         x = x.div(div)
#         return x


class AlexNet_Horse(nn.Module):

    def __init__(self):
        super(AlexNet_Horse, self).__init__()
        self.features = []
        self.features.append(nn.Conv2d(3, 96, 11, 4,))
        self.features.append(nn.ReLU(True))
        self.features.append(nn.CrossMapLRN2d(5))
        self.features.append(nn.MaxPool2d(3,2))
        
        self.features.append(nn.Conv2d(96, 256, 5, 1, 2,groups=2))
        self.features.append(nn.ReLU(True))
        self.features.append(nn.CrossMapLRN2d(5))
        self.features.append(nn.MaxPool2d(3,2))
        
        self.features.append(nn.Conv2d(256, 384, 3, 1, 1))
        self.features.append(nn.ReLU(True))
        
        self.features.append(nn.Conv2d(384, 384, 3, 1, 1,groups=2))
        self.features.append(nn.ReLU(True))
        
        self.features.append(nn.Conv2d(384, 256, 3, 1, 1,groups=2))
        self.features.append(nn.ReLU(True))
        self.features.append(nn.BatchNorm2d(256))

        self.features.append(nn.Conv2d(256, 128, 1))
        self.features.append(nn.ReLU(True))
        self.features.append(nn.MaxPool2d(3,2))

        self.features = nn.Sequential(*self.features)
        self.classifier = []
        self.classifier.append(nn.BatchNorm1d(4608))
        self.classifier.append(nn.Linear(4608,128))
        self.classifier.append(nn.ReLU(True))
        self.classifier.append(nn.Dropout(0.5))
        self.classifier.append(nn.BatchNorm1d(128))
        # self.classifier.append(nn.Linear(128,50))

        new_layer = nn.Linear(128,2)
        nn.init.xavier_normal(list(new_layer.parameters())[0])
        nn.init.constant(list(new_layer.parameters())[1],0.)
        self.classifier.append(new_layer)
        

        self.classifier = nn.Sequential(*self.classifier)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4608)
        x = self.classifier(x)
        return x

class Network:
    def __init__(self):
        model = load_lua('../data/pretrained_models/horse_tps_localization_minus_tps.t7')
        model_old = model.listModules()[1:]
        # for idx_layer_curr,layer_curr in enumerate(model_old):
        #     print idx_layer_curr, layer_curr
        # print nn.AvgPool3d
        
        model_py = AlexNet_Horse()
        for idx in range(18):
            for idx_param,param in enumerate(model_py.features[idx].parameters()):
                param.data = torch.FloatTensor(model_old[idx].parameters()[0][idx_param])
            
        for idx in range(5):
            # for idx_param,param in enumerate(model_py.features[idx].parameters()):
            #     print 'BEF',np.min(param.data.numpy()),np.max(param.data.numpy())
            
            for idx_param,param in enumerate(model_py.classifier[idx].parameters()):
                param.data = torch.FloatTensor(model_old[19+idx].parameters()[0][idx_param])
            
            # for idx_param,param in enumerate(model_py.features[idx].parameters()):
            #     print 'AFT',np.min(param.data.numpy()),np.max(param.data.numpy())
            
        # print model_py
        im = torch.FloatTensor(np.zeros((10,3,227,227)))
        # print im.shape
        out = model_py.forward(Variable(im))
        self.model = model_py

    def get_lr_list(self, lr):
        lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]\
        +[{'params': self.model.classifier[index].parameters(), 'lr': lr[1]} for index in range(len(self.model.classifier)-1)]\
        +[{'params': self.model.classifier[-1].parameters(), 'lr': lr[2]}]
        return lr_list
        # print out
        # .shape


# net = Network()
# print net.model
# print net.get_lr_list([0,0.01,0.001])
