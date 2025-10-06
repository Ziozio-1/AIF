import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from model import MNISTNet

def train(net, optimizer, loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total
	
if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default = 'MNIST', help='experiment name')
    parser.add_argument('--Batch_Size', type=int,default=1,help='Provide a specific batch Size. Default to 1.')
    parser.add_argument('--Learning_Rate', type=float, default=0.1, help='Provide the Learning rate of the model. Default to 0.1.')
    parser.add_argument('--Num_Epochs', type=int, default=10, help='Number of Epoch. Default to 10.')

    args = parser.parse_args()
    exp_name= args.exp_name
    epochs = args.Num_Epochs
    batch_size = args.Batch_Size
    lr = args.Learning_Rate

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

net = MNISTNet().to(device)
optimizer = optim.SGD(net.parameters(),lr=lr)

train(net, optimizer, trainloader, epochs=epochs)
test_acc = test(net,testloader)
print(f'Test accuracy:{test_acc}')

torch.save(net.state_dict(), 'weights/mnist_net.pth')