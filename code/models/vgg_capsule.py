from torchvision import models
import torch.nn as nn

import numpy as np
import scipy.misc
import torch
from CapsuleLayer import CapsuleLayer
from dynamic_capsules import Dynamic_Capsule_Model_Super
from spread_loss import Spread_Loss

class Vgg_Capsule(Dynamic_Capsule_Model_Super):

    def __init__(self,n_classes,pool_type='max',r=3,class_weights=None):
        super(Dynamic_Capsule_Model_Super, self).__init__()
        
        self.reconstruct = False
        if class_weights is not None:
            self.class_weights = torch.Tensor(class_weights[np.newaxis,:])

        if pool_type=='nopool':
            stride=2
        else:
            stride=1

        self.features = []
        self.features.append(nn.Conv2d(3, 64, 3, stride=stride,padding=1))
        self.features.append(nn.ReLU(True))
        if pool_type=='max':
            self.features.append(nn.MaxPool2d(2,2))
        elif pool_type=='avg':
            self.features.append(nn.AvgPool2d(2,2))
        
        self.features.append(nn.Conv2d(64, 128, 3, stride=stride,padding=1))
        self.features.append(nn.ReLU(True))
        if pool_type=='max':
            self.features.append(nn.MaxPool2d(2,2))
        elif pool_type=='avg':
            self.features.append(nn.AvgPool2d(2,2))
        
        self.features.append(CapsuleLayer(32, 1, 128, 8, kernel_size=7, stride=3, num_iterations=r))
        
        self.features.append(CapsuleLayer(n_classes, 32, 8, 16, kernel_size=6, stride=1, num_iterations=r))
        
        self.features = nn.Sequential(*self.features)
        
    # def forward(self, x):
    #     x = self.features(x)
    #     print x.size()
    #     return x

class Network:
    def __init__(self,n_classes=8,pool_type='max',r=3, init=False,class_weights = None):
        # print 'BN',bn
        # model = Vgg_Capsule(n_classes,pool_type,r,class_weights)
        # model_vgg = torch.load('../../data/vgg_face_torch/pytorch_vgg_face_just_conv.pth')
        import VGG_FACE

        model = VGG_FACE.VGG_FACE
        model.load_state_dict(torch.load('../../data/vgg_face_torch/VGG_FACE.pth'))

        print model

        # second_idx = 2 if pool_type=='nopool' else 3
        
        # # print model.features[0]
        # # print model.features[0].weight.data.shape
        # # print model.features[second_idx]
        # # print model.features[second_idx].weight.data.shape
        # # print model_vgg
        # # print model_vgg[0].weight.data.shape
        # # print model_vgg[5].weight.data.shape

        # model.features[0].weight.data = model_vgg[0].weight.data
        # model.features[0].bias.data = model_vgg[0].bias.data

        # model.features[second_idx].weight.data = model_vgg[5].weight.data
        # model.features[second_idx].bias.data = model_vgg[5].bias.data
        

        # if init:
        #     for idx_m,m in enumerate(model.features):
        #         if isinstance(m, CapsuleLayer):
        #             # print m,2
        #             if m.num_in_capsules==1:
        #                 # print m,3
        #                 nn.init.xavier_normal(m.capsules.weight.data)
        #                 nn.init.constant(m.capsules.bias.data,0.)
        #             else:
        #                 # print m,4
        #                 nn.init.normal(m.route_weights.data, mean=0, std=0.1)
                
        self.model = model
        
    
    def get_lr_list(self, lr):
        lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]
        # \
        #         +[{'params': self.model.classifier.parameters(), 'lr': lr[1]}]
        return lr_list


def saving_vgg_conv():
    import VGG_FACE

    model = VGG_FACE.VGG_FACE
    print list(model.children())[0].weight.data[0,0]
    model.load_state_dict(torch.load('../../data/vgg_face_torch/VGG_FACE.pth'))
    # print model
    print list(model.children())[0].weight.data[0,0]

    conv_layers = list(model.children())
    conv_layers = conv_layers[:30]
    model = nn.Sequential(*conv_layers)
    
    print list(model.children())[0].weight.data[0,0]

    # torch.save(model,'./pytorch_vgg_face_just_conv_no_weights.pth')


def main():
    # saving_vgg_conv()
    # return
    import numpy as np
    import torch
    from torch.autograd import Variable
    from torchvision import transforms    

    net = Network(n_classes= 8, pool_type='max', init = True)
    net.model = net.model.cuda()
    image_path = '../../data/vgg_face_torch/ak.png'
    im = scipy.misc.imread(image_path)
    im = im[:,:,[2,1,0]]
    print im.shape,np.min(im),np.max(im)

    net.model.eval()
    # mean_std = np.array([[129.1863,104.7624,93.5940],[1.,1.,1.]]) #rgb
    mean_std = np.array([[93.5940,104.7624,129.1863],[1.,1.,1.]]) #bgr
    
    print mean_std

    trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            # transforms.RandomCrop(im_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            lambda x: x*255,
            transforms.Normalize(mean_std[0,:],mean_std[1,:]),
            # lambda x: x[[2, 1, 0],:, : ]
        ])
    im_t = trans(im)
    im_t = im_t.view(1,im_t.size(0),im_t.size(1),im_t.size(2))

    print im_t.size(),torch.min(im_t),torch.max(im_t)
    out = net.model(Variable(im_t.cuda()))
    print out.size()
    print torch.max(out,1)
    print out.data[0,:10]
    

    # print net.model
    # labels = np.ones((10,))
    # net.model = net.model.cuda()
    # input = np.zeros((10,3,96,96))
    # input = torch.Tensor(input).cuda()
    # print input.shape
    # input = Variable(input)
    # labels = Variable(torch.LongTensor(labels).cuda())
    # output = net.model(input)
    # print output.data.shape

    # criterion = Spread_Loss(50,5)
    # for epoch_num in range(53):
    #     print epoch_num,criterion(output,labels,epoch_num)



if __name__=='__main__':
    main()



