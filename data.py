import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


basicTransform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


augmentedTransform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomHorizontalFlip,
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Source: https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
