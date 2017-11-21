from torchvision import models
import torch.nn as nn


class Network:
    def __init__(self):
        model_ft = models.alexnet(pretrained=True)
        new_classifier = list(model_ft.classifier.children())
        new_classifier.pop()
        new_layer = nn.Linear(4096,2)
        nn.init.xavier_normal(list(new_layer.parameters())[0])
        nn.init.constant(list(new_layer.parameters())[1],0.)
        new_classifier.append(new_layer)
        model_ft.classifier = nn.Sequential(*new_classifier)
        self.model = model_ft

    def get_lr_list(self, lr):
        lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]\
        +[{'params': self.model.classifier[index].parameters(), 'lr': lr[1]} for index in range(len(self.model.classifier)-1)]\
        +[{'params': self.model.classifier[-1].parameters(), 'lr': lr[2]}]
        return lr_list