import torch.utils
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import h5py

def get_alexnet():
	model = mode

def main():
	data_transforms = {
	    'train': transforms.Compose([
	        transforms.RandomSizedCrop(224),
	        transforms.RandomHorizontalFlip(),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ]),
	    'val': transforms.Compose([
	        transforms.Scale(256),
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ]),
	}

	model_ft = models.alexnet(pretrained=True)
	new_classifier = list(model_ft.classifier.children())
	new_classifier.pop()
	new_classifier.append(nn.Linear(4096, 2))
	model_ft.classifier = nn.Sequential(*new_classifier)

	criterion = nn.CrossEntropyLoss()

	# # Observe that all parameters are being optimized
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

	# # Decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

	data_files = ['../data/horse_51/train_test_split/train_0.h5','../data/horse_51/train_test_split/test_0.h5']

	file    = h5py.File(train_data_file, 'r')   # 'r' means that hdf5 file is open in read-only mode
	train_data = torch.utils.data.DataLoader(zip(file['data'],file['labels']),batchsize=4)
	


if __name__=='__main__':
	main()