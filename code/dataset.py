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


