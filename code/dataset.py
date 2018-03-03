import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from helpers import util, visualize
import scipy.misc
from PIL import Image


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

        # print np.min(image),np.max(image)
        image = image-self.mean
        # print np.min(image),np.max(image)
        image = image/self.std
        # print np.min(image),np.max(image)
        # image = image- self.mean
        image = image[:,:,np.newaxis]
        
        # print np.min(image), np.max(image)        
        
        sample = {'image': image, 'label': label}
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

