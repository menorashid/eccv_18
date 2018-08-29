import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable
import torchvision.models as models

def main():
    vgg16 = models.vgg16()

    vgg_base = list(vgg16.features.children())
    vgg_base = vgg_base[:30]
    vgg_base = nn.Sequential(*vgg_base)
    torch.save(vgg_base, 'pytorch_vgg_imagenet_just_conv.pth')

    # print model
    # print vgg_base
    # vgg_base = torch.load('pytorch_vgg_face_just_conv.pth')
    # print vgg_base
    # print vgg16

if __name__=='__main__':
    main()