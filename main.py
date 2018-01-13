import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from data import classes, basicTransform
from train_and_eval import train, evaluate
from model import CNN, VGGLike, ResidualNet

'''Setup command line arguments'''
parser = argparse.ArgumentParser(description='CNN tutorial')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate (default: 0.0005)')
parser.add_argument('--bs', type=int, default=32,
                    help='batch size (default 32)')
parser.add_argument('--epochs', type=int, default=30,
                    help='number epochs (default 10)')
parser.add_argument('--cuda', type=int, default=1,
                    help='whether to use cuda')
parser.add_argument('--device', type=int, default=0,
                    help='select GPU device')
parser.add_argument('--nettype', type=int, default=0,
                    help='which type of CNN to use (default = 0: basic, 1 = VGG like, 2 = ResNet)')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout percentage (default: 0.3)')
parser.add_argument('--filters', type=int, default=32,
                    help='initial number of filters for VGG like net and ResNets (default: 32)')
parser.add_argument('--model_path', type=str, default='./models/',
                    help='directory to save models in')
parser.add_argument('--data_path', type=str, default='./data',
                    help='directory to store data in')
args = parser.parse_args()
print(args)

'''Setup hyperparameters from command line options'''
batch_size = args.bs
num_epochs = args.epochs
learning_rate = args.lr
cuda = args.cuda
cuda_device = args.device
dropout = args.dropout
filters = args.filters
model_path = args.model_path
data_path = args.data_path
transform = basicTransform
nclasses = 10

'''Load training and test data'''
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

'''Initialize network, optimizer, and loss function'''
if args.nettype == 0:
    print("Model is a basic CNN")
    net = CNN(dropout, nclasses)
    name = 'basicCNN'
elif args.nettype == 1:
    print("Model is a VGG like CNN")
    in_dim = (3, 32, 32)
    net = VGGLike(in_dim, filters, dropout, nclasses)
    name = 'VGGlike'
elif args.nettype == 2:
    in_dim = (3, 32, 32)
    channels = 3
    num_res_blocks = 4
    net = ResidualNet(in_dim,
                      channels,
                      filters,
                      num_res_blocks,
                      dropout,
                      nclasses)
    name = 'ResNet'
else:
    print("Unknown net type")
    sys.exit()
print(net)
optimizer = optim.Adam(net.params, lr=learning_rate)
criterion = nn.CrossEntropyLoss()

'''Convert to cuda if available'''
if torch.cuda.is_available() and cuda:
    print("CUDA is available, training on GPU")
    print("Number of available devices: {}".format(torch.cuda.device_count()))
    print("Using device: {}".format(cuda_device))
    torch.cuda.device(args.device)
    net.cuda()
    criterion = criterion.cuda()
else:
    print("CUDA is NOT available, training on CPU")

'''Train and evaluate model'''
for i in range(1, num_epochs + 1):
    train(i, net, trainloader, criterion, optimizer, cuda, batch_size)
    evaluate(i, net, testloader, criterion, cuda, batch_size)
    torch.save(net.state_dict(), model_path + name + "_" + str(i) + ".pth")
    if i % 2 == 0:
        '''Decay learning rate'''
        learning_rate = learning_rate * 0.95
        optimizer = optim.Adam(net.params, lr=learning_rate)
    print(f"Learning rate: {learning_rate}")
