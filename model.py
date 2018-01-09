import torch
import torch.nn as nn
from torch.autograd import Variable


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
        layers += [nn.ReLU()]
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
    Dropout before or after max pooling?
    https://stats.stackexchange.com/questions/147850/are-pooling-layers-added-before-or-after-dropout-layers
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
        layers += [nn.ReLU()]
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
