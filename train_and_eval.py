import torch
from torch.autograd import Variable


def train(epoch, net, dataloader, criterion, optimizer, cuda, batch_size):
    net.train()
    correct = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        if data.size(0) != batch_size:
            continue
        data, target = Variable(data), Variable(target)
        if torch.cuda.is_available() and cuda:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        '''Keep track of accuracy and loss'''
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        total_loss += loss.data[0]
        '''Report progress'''
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.data[0]))
    total_loss = (total_loss * batch_size) / len(dataloader.dataset)
    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        total_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))


def evaluate(epoch, net, dataloader, criterion, optimizer, cuda, batch_size):
    net.eval()
    correct = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        if data.size(0) != batch_size:
            continue
        data, target = Variable(data), Variable(target)
        if torch.cuda.is_available() and cuda:
            data = data.cuda()
            target = target.cuda()
        output = net(data)
        loss = criterion(output, target)
        '''Keep track of accuracy and loss'''
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        total_loss += loss.data[0]
    total_loss = (total_loss * batch_size) / len(dataloader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        total_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))
