import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def init_layers(params, layer_type):
    '''
    Initializes all of the layers of type 'Linear' or 'Conv' using xavier uniform initialization for the weights and 0.01 for the biases
    Initializes all layers of type 'BatchNorm' using uniform initialization for the weights and the same as above for the biases
    '''
    biasinit = 0.01
    for p in params:
        classname = p.__class__.__name__
        if classname.find(layer_type) != -1:
            if layer_type == 'BatchNorm':
                torch.nn.init.uniform(layer.weight.data)
            else:
                torch.nn.init.xavier_uniform(layer.weight.data)
            layer.bias.data.fill_(biasinit)


class CNN(nn.Module):
    '''Simple convolutional neural network, with 2 conv layers and one fully connected layer'''
    def __init__(self, dropout, nclasses):
        super(CNN, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=0, dilation=1)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0, dilation=1)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout2d(dropout)]
        self.conv_model = nn.Sequential(*layers)
        layers = []
        layers += [nn.Linear(in_features=800, out_features=nclasses)]
        self.flat_model = nn.Sequential(*layers)
        self.params = list(self.conv_model.parameters()) + list(self.flat_model.parameters())
        init_layers(self.params, 'Conv')
        init_layers(self.params, 'Linear')

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(-1, 800)
        x = self.flat_model(x)
        return x

    def __str__(self):
        return self.conv_model.__str__() + '\n' + self.flat_model.__str__()


class VGGLike(nn.Module):
    '''
    Original paper: https://arxiv.org/pdf/1409.1556.pdf
    '''

    def __init__(self, in_dim, filt, drop_p, nclasses):
        super(VGGLike, self).__init__()
        self.in_dim = in_dim
        layers = []
        # Block 1
        layers += [nn.Conv2d(3, filt, kernel_size=3)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(filt, filt, kernel_size=3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(2)]
        layers += [nn.Dropout2d(drop_p)]
        # Block 2
        layers += [nn.Conv2d(filt, filt * 2, kernel_size=3)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(filt * 2, filt * 2, kernel_size=3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(2)]
        layers += [nn.Dropout2d(drop_p)]
        # Block 3
        layers += [nn.Conv2d(filt * 2, filt * 4, kernel_size=3)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(filt * 4, filt * 4, kernel_size=3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout2d(drop_p)]
        self.conv_model = nn.Sequential(*layers)
        # Dense model
        self.num_flat_feats = self.get_conv_output_size()
        layers = []
        layers += [nn.Linear(in_features=self.num_flat_feats, out_features=nclasses)]
        self.flat_model = nn.Sequential(*layers)
        self.params = list(self.conv_model.parameters()) + list(self.flat_model.parameters())
        init_layers(self.params, 'Conv')
        init_layers(self.params, 'Linear')

    def get_conv_output_size(self):
        x = Variable(torch.ones(1, *self.in_dim))
        x = self.conv_model(x)
        return x.numel()

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(-1, self.num_flat_feats)
        x = self.flat_model(x)
        return x

    def __str__(self):
        return self.conv_model.__str__() + '\n' + self.flat_model.__str__()


'''
Residual network
Original paper: Deep Residual Learning for Image Recognition https://arxiv.org/pdf/1512.03385.pdf
Also, inspiration from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
And here for how incorporate a varying number of residual blocks, and contain the blocks
nicely
https://github.com/mingyuliutw/UNIT
'''


class ResidualBlock(nn.Module):
    def __init__(self, filt):
        super(ResidualBlock, self).__init__()
        model = []
        model += [nn.Conv2d(filt, filt, kernel_size=3, stride=1, padding=1)]
        model += [nn.BatchNorm2d(filt)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Conv2d(filt, filt, kernel_size=3, stride=1, padding=1)]
        model += [nn.BatchNorm2d(filt)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out += x
        return F.relu(out)


class ResidualNet(nn.Module):
    def __init__(self,
                 in_dim,
                 channels,
                 filt,
                 num_res_blocks,
                 drop_p,
                 nclasses):
        super(ResidualNet, self).__init__()
        self.in_dim = in_dim
        model = []
        # Large kernel and stride of 2 so that image shrunk to filt * 14 * 14 dim
        model += [nn.Conv2d(channels, filt, kernel_size=5, stride=2)]
        model += [nn.BatchNorm2d(filt)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Dropout(drop_p)]
        # Add residual blocks. Downsample size and double number of filters every 2 blocks
        for i in range(num_res_blocks):
            if (i + 1) % 2 == 0:
                model += [ResidualBlock(filt)]
                model += [nn.Conv2d(filt, filt * 2,
                                    kernel_size=3, stride=2, padding=1)]
                filt = filt * 2
                model += [nn.BatchNorm2d(filt)]
                model += [nn.ReLU(inplace=True)]
                model += [nn.Dropout(drop_p)]
            else:
                model += [ResidualBlock(filt)]
                model += [nn.Dropout(drop_p)]
        self.res_blocks = nn.Sequential(*model)
        self.flat_weights = self.get_conv_output_size()
        print("Number of flat weights: {}".format(self.flat_weights))
        self.fc1 = nn.Linear(self.flat_weights, nclasses)
        self.params = list(self.res_blocks.parameters())
        # Initialize model parameters
        init_layers(self.params, 'Linear')
        init_layers(self.params, 'Conv')
        init_layers(self.params, 'BatchNorm')
        torch.nn.init.xavier_uniform(self.fc1.weight.data)
        self.fc1.bias.data.fill_(0.01)

    def get_conv_output_size(self):
        x = Variable(torch.ones(1, *self.in_dim))
        x = self.res_blocks(x)
        return x.numel()

    def forward(self, x):
        x = self.res_blocks(x)
        x = x.view(-1, self.flat_weights)
        x = self.fc1(x)
        return x

    def __str__(self):
        return self.res_blocks.__str__() + '\n' + self.fc1.__str__()
