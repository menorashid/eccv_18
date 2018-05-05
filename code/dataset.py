import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from helpers import util, visualize
import scipy.misc
from PIL import Image
import cv2

class generic_dataset(Dataset):
    def __init__(self, text_file, transform=None):
        self.files = util.readLinesFromFile(text_file)
        self.transform = transform
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        train_file_curr,label = train_file_curr.split(' ')
        label = int(label)
        image = Image.open(train_file_curr)
        sample = {'image': image, 'label': label}
        # if self.transform:
        sample['image'] = self.transform(sample['image'])

        return sample

class Horse_Image_Dataset(generic_dataset):
    def __init__(self, text_file, transform=None):
        super(Horse_Image_Dataset, self).__init__(text_file,transform)
    

class CK_96_Dataset(generic_dataset):
    def __init__(self, text_file, mean_file, std_file, transform=None):
        super(CK_96_Dataset, self).__init__(text_file,transform)
        self.mean = scipy.misc.imread(mean_file).astype(np.float32)
        self.std = scipy.misc.imread(std_file).astype(np.float32)
        self.std[self.std==0]=1.

        
    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        train_file_curr,label = train_file_curr.split(' ')
        label = int(label)
        image = scipy.misc.imread(train_file_curr).astype(np.float32)
        if image.shape[0]!=96 or image.shape[1]!=96:
            image = scipy.misc.imresize(image,(96,96)).astype(np.float32)

        # print 'new_im'
        # print np.min(image),np.max(image)
        image = image-self.mean
        # print np.min(image),np.max(image),np.min(self.mean),np.max(self.mean)
        image = image/self.std
        # print np.min(image),np.max(image),np.min(self.std),np.max(self.std)
        image = image[:,:,np.newaxis]
        
        # print np.min(image), np.max(image)        
        # raw_input()
        
        sample = {'image': image, 'label': label}
        sample['image'] = self.transform(sample['image'])

        return sample

class CK_96_Dataset_with_rs(generic_dataset):
    def __init__(self, text_file, mean_file, std_file, transform=None,resize = None):
        super(CK_96_Dataset_with_rs, self).__init__(text_file,transform)
        self.resize = resize
        self.mean = scipy.misc.imread(mean_file)
        self.std = scipy.misc.imread(std_file)
        self.std[self.std==0]=1.
        
        if self.resize is not None:
            self.mean = scipy.misc.imresize(self.mean,(self.resize,self.resize))
            self.std = scipy.misc.imresize(self.std,(self.resize,self.resize))
        
        self.mean = self.mean.astype(np.float32)
        self.std = self.std.astype(np.float32)

    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        train_file_curr,label = train_file_curr.split(' ')
        label = int(label)
        image = scipy.misc.imread(train_file_curr)
        if self.resize is not None:
            if image.shape[0]!= self.resize or image.shape[1]!= self.resize:
                image = scipy.misc.imresize(image,(self.resize,self.resize))

        image = image.astype(np.float32)

        # print 'new_im'
        # print np.min(image),np.max(image)
        image = image-self.mean
        # print np.min(image),np.max(image),np.min(self.mean),np.max(self.mean)
        image = image/self.std
        # print np.min(image),np.max(image),np.min(self.std),np.max(self.std)
        image = image[:,:,np.newaxis]
        
        # print np.min(image), np.max(image)        
        # raw_input()
        
        sample = {'image': image, 'label': label}
        sample['image'] = self.transform(sample['image'])

        return sample

class CK_96_Dataset_WithAU(generic_dataset):
    def __init__(self, text_file, mean_file, std_file, transform=None):
        super(CK_96_Dataset_WithAU, self).__init__(text_file,transform)
        self.mean = scipy.misc.imread(mean_file).astype(np.float32)
        self.std = scipy.misc.imread(std_file).astype(np.float32)
        self.std[self.std==0]=1.
        
    def __getitem__(self, idx):
        line_curr = self.files[idx]
        
        line_split = line_curr.split(' ')
        train_file_curr = line_split[0]
        annos= [int(val) for val in line_split[1:]]
        label = annos[0]
        bin_au = annos[1]
        label_au = np.array(annos[2:])

        image = scipy.misc.imread(train_file_curr).astype(np.float32)
        if image.shape[0]!=96 or image.shape[1]!=96:
            image = scipy.misc.imresize(image,(96,96)).astype(np.float32)

        image = image-self.mean
        image = image/self.std
        image = image[:,:,np.newaxis]
        
        
        sample = {'image': image, 'label': label, 'label_au': label_au, 'bin_au':bin_au}
        sample['image'] = self.transform(sample['image'])

        return sample

class CK_96_Dataset_Just_Mean(generic_dataset):
    def __init__(self, text_file, mean_file, std_file, transform=None):
        super(CK_96_Dataset_Just_Mean, self).__init__(text_file,transform)
        self.mean = scipy.misc.imread(mean_file).astype(np.float32)
        # self.std = scipy.misc.imread(std_file).astype(np.float32)
        # self.std[self.std==0]=1.
        
    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        train_file_curr,label = train_file_curr.split(' ')
        label = int(label)
        image = scipy.misc.imread(train_file_curr).astype(np.float32)
        # print np.min(image),np.max(image)
        image = image-self.mean
        # print np.min(image),np.max(image)
        # image = image/self.std
        # print np.min(image),np.max(image)
        # image = image- self.mean
        image = image[:,:,np.newaxis]
        
        # print np.min(image), np.max(image)        
        
        sample = {'image': image, 'label': label}
        sample['image'] = self.transform(sample['image'])

        return sample

class CK_96_New_Dataset(generic_dataset):
    def __init__(self, text_file, transform=None):
        super(CK_96_New_Dataset, self).__init__(text_file,transform)
        
    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        train_file_curr,label = train_file_curr.split(' ')
        label = int(label)
        image = scipy.misc.imread(train_file_curr)
        # .astype(np.float32)
        # print np.min(image),np.max(image)
        # image = image-self.mean
        # # print np.min(image),np.max(image)
        # image = image/self.std
        # # print np.min(image),np.max(image)
        # # image = image- self.mean
        image = image[:,:,np.newaxis]
        
        # print np.min(image), np.max(image)        
        
        sample = {'image': image, 'label': label}
        sample['image'] = self.transform(sample['image'])

        return sample

class Oulu_Static_Dataset(generic_dataset):
    def __init__(self, text_file, transform=None, bgr = False,color=False):
        super(Oulu_Static_Dataset, self).__init__(text_file,transform)
        self.bgr = bgr
        self.color = color
        
    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        train_file_curr,label = train_file_curr.split(' ')
        label = int(label)
        image = scipy.misc.imread(train_file_curr)
        
        if len(image.shape)==2:
            image = image[:,:,np.newaxis]
            if self.color:
                image = np.concatenate((image,image,image),2)

        if self.bgr:
            image = image[:,:,[2,1,0]]
        
        sample = {'image': image, 'label': label}
        sample['image'] = self.transform(sample['image'])

        return sample


class CK_for_VGG(Oulu_Static_Dataset):
    def __init__(self, text_file, transform=None, bgr = False,color=False):
        super(Oulu_Static_Dataset, self).__init__(text_file,transform,bgr,color)
    


class Bp4d_Dataset(generic_dataset):
    def __init__(self, text_file, bgr = False, binarize = False, transform=None):
        # super(Disfa_10_6_Dataset, self).__init__(text_file,transform)
        self.bgr = bgr
        self.files = util.readLinesFromFile(text_file)
        # [:1280]
        self.binarize = binarize
        self.transform = transform
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        
    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        info = train_file_curr.split(' ')
        train_file_curr = info[0]
        labels = [int(val) for val in info[1:]]
        labels = np.array(labels)
        # print labels
        
        if self.binarize :
            labels[labels>0]=1
            labels[labels<1]=0

        # .astype('float')
        # labels[labels>0]=1
        # labels[labels<1]=0
        # labels[labels<0.5]=0.1


        
        if self.bgr:
            image = cv2.imread(train_file_curr)
            # .astype(float32)
            # print image.shape,type(image)
            # image = scipy.misc.imread(train_file_curr)          
            # print image.shape,type(image)
            # raw_input()
        else:
            image = scipy.misc.imread(train_file_curr)          


        sample = {'image': image, 'label': labels}
        sample['image'] = self.transform(sample['image'])

        return sample

class Bp4d_Dataset_with_mean_std_val(generic_dataset):

    def __init__(self, text_file, mean_std = None, resize = None, binarize = False, transform=None, bgr= False):
        super(Bp4d_Dataset_with_mean_std_val, self).__init__(text_file,transform)
        # self.mean = scipy.misc.imread(mean_file)
        # .astype(np.float32)
        # self.std = scipy.misc.imread(std_file)
        # .astype(np.float32)
        # self.std[self.std==0]=1.
        self.resize = resize
        self.binarize = binarize
        self.bgr = bgr

        # if self.resize is not None:
            # self.mean = scipy.misc.imresize(self.mean,(self.resize,self.resize))
            # .astype(np.float32)
            # self.std = scipy.misc.imresize(self.std,(self.resize,self.resize))
            # .astype(np.float32)
        # print np.min(self.mean),np.max(self.mean)
        # print np.min(self.std),np.max(self.std)

        self.mean = mean_std[0]
        self.mean = self.mean[np.newaxis,np.newaxis,:]
        self.std = mean_std[1]
        self.std = self.std[np.newaxis,np.newaxis,:]
        
        # self.std = self.std.astype(np.float32)

        # print np.min(self.mean),np.max(self.mean)
        # print np.min(self.std),np.max(self.std)

        # raw_input()


        
    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        info = train_file_curr.split(' ')
        train_file_curr = info[0]
        labels = [int(val) for val in info[1:]]
        labels = np.array(labels)

        # .astype('float')
        if self.binarize :
            # print labels
            labels[labels>0]=1
            labels[labels<1]=0
        # labels[labels<0.5]=0.1
        image = scipy.misc.imread(train_file_curr)
        # print image.shape
        if self.bgr:
            image = image[:,:,[2,1,0]]        
        
        if self.resize is not None:
            if image.shape[0]!= self.resize or image.shape[1]!= self.resize:
                image = scipy.misc.imresize(image,(self.resize,self.resize))
        image = image.astype(np.float32)          

        image = image-self.mean
        image = image/self.std
        
        # print 'hello'
        # print image.shape
        # print image.shape, np.min(np.min(image,0),0),np.max(np.max(image,0),0)
        # print self.mean.shape
        # print self.std.shape
        # print image.shape, np.min(np.min(image,0),0),np.max(np.max(image,0),0)
        # raw_input()
        sample = {'image': image, 'label': labels}
        sample['image'] = self.transform(sample['image'])

        return sample

class Bp4d_Dataset_Mean_Std_Im(generic_dataset):

    def __init__(self, text_file, mean_file, std_file, resize = None, binarize = False, transform=None):
        super(Bp4d_Dataset_Mean_Std_Im, self).__init__(text_file,transform)
        self.mean = scipy.misc.imread(mean_file)
        # .astype(np.float32)
        self.std = scipy.misc.imread(std_file)
        # .astype(np.float32)
        self.std[self.std==0]=1.
        self.resize = resize
        self.binarize = binarize

        if self.resize is not None:
            self.mean = scipy.misc.imresize(self.mean,(self.resize,self.resize))
            # .astype(np.float32)
            self.std = scipy.misc.imresize(self.std,(self.resize,self.resize))
            # .astype(np.float32)
        # print np.min(self.mean),np.max(self.mean)
        # print np.min(self.std),np.max(self.std)

        self.mean = self.mean.astype(np.float32)
        self.std = self.std.astype(np.float32)

        # print np.min(self.mean),np.max(self.mean)
        # print np.min(self.std),np.max(self.std)

        # raw_input()


        
    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        info = train_file_curr.split(' ')
        train_file_curr = info[0]
        labels = [int(val) for val in info[1:]]
        labels = np.array(labels)

        # .astype('float')
        if self.binarize :
            # print labels
            labels[labels>0]=1
            labels[labels<1]=0
        # labels[labels<0.5]=0.1
        image = scipy.misc.imread(train_file_curr)
        # print image.shape

        if self.resize is not None:
            if image.shape[0]!= self.resize or image.shape[1]!= self.resize:
                image = scipy.misc.imresize(image,(self.resize,self.resize))
        image = image.astype(np.float32)          
        # print image.shape
        # print self.mean.shape
        # print self.std.shape

        image = image-self.mean
        # print np.min(image),np.max(image),np.min(self.mean),np.max(self.mean)
        image = image/self.std
        # print np.min(image),np.max(image),np.min(self.std),np.max(self.std)
        image = image[:,:,np.newaxis]
        
        # print np.min(image), np.max(image)        
        # print labels
        # print type(labels[0])

        # raw_input()
        
        sample = {'image': image, 'label': labels}
        sample['image'] = self.transform(sample['image'])

        return sample



class Disfa_10_6_Dataset(generic_dataset):
    def __init__(self, text_file, bgr = False, transform=None):
        # super(Disfa_10_6_Dataset, self).__init__(text_file,transform)
        self.bgr = bgr
        self.files = util.readLinesFromFile(text_file)[:1]
        self.transform = transform
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        
    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        info = train_file_curr.split(' ')
        train_file_curr = info[0]
        labels = [int(val) for val in info[1:]]
        labels = np.array(labels)
        # .astype('float')
        labels[labels>0]=1
        labels[labels<1]=0
        # labels[labels<0.5]=0.1


        image = scipy.misc.imread(train_file_curr)
        if self.bgr:
            image = image[:,:,[2,1,0]]        
        sample = {'image': image, 'label': labels}
        sample['image'] = self.transform(sample['image'])

        return sample


class CK_RS_Dataset(generic_dataset):
    def __init__(self, text_file, mean_file, std_file, im_size, transform=None):
        super(CK_RS_Dataset, self).__init__(text_file,transform)
        self.im_size = im_size
        self.mean = scipy.misc.imresize(scipy.misc.imread(mean_file),(im_size,im_size)).astype(np.float32)
        self.std = scipy.misc.imresize(scipy.misc.imread(std_file),(im_size,im_size)).astype(np.float32)
        self.std[self.std==0]=1.
        
    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        train_file_curr,label = train_file_curr.split(' ')
        label = int(label)
        image = scipy.misc.imresize(scipy.misc.imread(train_file_curr),(self.im_size,self.im_size)).astype(np.float32)
        # print np.min(image),np.max(image)
        image = image-self.mean
        # print np.min(image),np.max(image)
        image = image/self.std
        # print np.min(image),np.max(image)
        # image = image- self.mean
        image = image[:,:,np.newaxis]
        
        
        sample = {'image': image, 'label': label}
        sample['image'] = self.transform(sample['image'])

        return sample

