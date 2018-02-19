from helpers import util,visualize,augmenters
from torchvision import datasets, transforms
import dataset
import torch
import numpy as np
import scipy.misc

def main():
    split_num = 0
    train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
    test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
    mean_std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean_std_val_0_1.npy'

    mean_std = np.load(mean_std_file)

    print mean_std

    batch_size_val = 4
    data_transforms = {}
    data_transforms['train']= transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(66),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(64),
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
    ])
    data_transforms['val']= transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([float(mean_std[0])],[float(mean_std[1])])
        ])

    train_data = dataset.CK_96_New_Dataset(train_file, data_transforms['train'])
    test_data = dataset.CK_96_New_Dataset(test_file, data_transforms['val'])
    

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_val,
                        shuffle=False, num_workers=1)

    for num_iter,batch in enumerate(test_dataloader):
                
        # batch = test_dataloader.next() 
        ims = batch['image'].numpy()
        # ims = ims.numpy()
        print np.min(ims),np.max(ims),ims.shape

        break

    #     # labels_all.append(batch['label'].numpy())
    

if __name__=='__main__':
    main()
