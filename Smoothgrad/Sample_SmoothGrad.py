# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 22:23:54 2023

@author: ianni
"""

import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image


    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels

    import torch.nn as nn
    import torch.nn.functional as F


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    PATH = './cifar_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))


    ########### EVERYTHING ABOVE IS MODEL TRAINING ###########

    # Gradient Generating Method
    def returnGrad(img, model, criterion, device = 'cpu'):
        model.to(device)
        img = img.to(device)
        img.requires_grad_(True).retain_grad()
        pred = model(img)
        loss = criterion(pred, torch.tensor([int(torch.max(pred[0], 0)[1])]).to(device))
        loss.backward()
        
    #    S_c = torch.max(pred[0].data, 0)[0]
        Sc_dx = img.grad
        
        return Sc_dx

    # img = images[0:1]
    # Extract Gradient from Input and Plot
    # gradient = returnGrad(img = img, model = net, criterion = criterion)
    # imshow(torchvision.utils.make_grid(gradient))
    
    # Gradient Generating Method for SmoothGrad
    def returnSmoothGrad(img, model, criterion, sigma, device = 'cpu'):
        N = 10
        sg_total = torch.zeros_like(img)
        for i in range(N):
            noise = torch.tensor(np.random.normal(0, sigma, img.shape), dtype=torch.float)
            noise_img = img + noise
            sg_total += returnGrad(img=noise_img, model=model, criterion=criterion)
        return sg_total
        
    img = images[0:1]
    
    # Extract SmoothGrad from Input and Plot
    smooth_gradient = returnSmoothGrad(img=img, model = net, criterion = criterion, sigma=0.02)
    imshow(torchvision.utils.make_grid(smooth_gradient))
    imshow(torchvision.utils.make_grid(img))
