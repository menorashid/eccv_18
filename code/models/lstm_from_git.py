import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
print model

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.squeeze().to(device)
        # reshape(-1, sequence_length, input_size).to(device)
        # print images.size()
        # raw_input()
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
# import torch
# import argparse

# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F

# from tqdm import tqdm

# from torch import optim
# from torch.autograd import Variable

# from torchvision import datasets, transforms

# parser = argparse.ArgumentParser(description='Recurrent Unit Baselines')

# parser.add_argument('--batch_size', help='batch size of network', type=int, default=16)
# parser.add_argument('--epochs', help='number of epochs', type=int, default=10)
# parser.add_argument('--hidden_layer_size', help='size of the hidden layer', type=int, default=100)
# parser.add_argument('--gpu', help='use gpu for training', action='store_true')
# parser.add_argument('--learning_rate', help='the learning rate', type=float, default=0.01)
# parser.add_argument('--gradient_clipping_value', help='the gradient clipping value', type=int, default=1)

# args = parser.parse_args()


# def sequential_MNIST(batch_size, gpu=False, dataset_folder='./data'):
#     kwargs = {'num_workers': 1, 'pin_memory': True} if gpu else {}

#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST(dataset_folder, train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,)),
#                            transforms.Lambda(lambda x: x.view(-1, 1))
#                        ])),
#         batch_size=batch_size, shuffle=True, **kwargs)

#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST(dataset_folder, train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,)),
#             transforms.Lambda(lambda x: x.view(-1, 1))
#         ])),
#         batch_size=batch_size, shuffle=False, **kwargs)

#     return (train_loader, test_loader)


# training_data, testing_data = sequential_MNIST(args.batch_size, gpu=args.gpu)


# class LSTMBaseline(nn.Module):

#     def __init__(self):
#         super(LSTMBaseline, self).__init__()
#         self.input_layer = nn.LSTM(1, args.hidden_layer_size, batch_first=True)
#         self.linear_layer = nn.Linear(args.hidden_layer_size, 10)

#     def forward(self, x):
#         x, _ = self.input_layer(x)
#         x = self.linear_layer(x[:, -1, :])
#         return x

# model = LSTMBaseline()

# if args.gpu:
#     model.cuda()

# criterion = nn.CrossEntropyLoss()


# def train():
#     model.train()

#     for current_batch, (data, label) in enumerate(tqdm(training_data)):

#         if args.gpu:
#             data, label = Variable(data).cuda(), Variable(label).cuda()
#         else:
#             data, label = Variable(data), Variable(label)

#         model.zero_grad()
#         output = model(data)

#         loss = criterion(output, label)

#         if (current_batch + 1) % 100 == 0:
#             print('Current Loss: {%.2f}' % loss)

#         loss.backward()

#         torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clipping_value)

#         for p in model.parameters():
#             p.data.add_(-args.learning_rate, p.grad.data)


# def test():
#     model.eval()

#     correct = 0
#     total = 0

#     print('Testing accurracy...')

#     for data, label in tqdm(testing_data):
#         total += label.size(0)

#         if args.gpu:
#             data, label = Variable(data).cuda(), Variable(label).cuda()
#         else:
#             data, label = Variable(data), Variable(label)

#         output = model(data)
#         _, predicted = torch.max(output.data, 1)

#         correct += (predicted == label.data).sum()

#     print(str(100 * correct / total) + '%')


# def main():
#     # print "HELLOOOOO"
#     for epoch in range(1, args.epochs):
#         print('Epoch: {%d}'%epoch)
#         train()
#         test()


# if __name__ == '__main__':
#     main()