import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import dataset
import resnet
from dataset import label_list

def train(net, epoch_num, device):
    print('#start train#')
    net.train()
    for epoch_id in tqdm(range(epoch_num)):
        running_loss = 0.0
        for j,data in tqdm(enumerate(dataset.train_loader)):
            inputs,labels = data

            inputs,labels = inputs.to(device), labels.to(device)    #把数据放到GPU/CPU上
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print('epoch: %d, loss: %.4f.' % (epoch_id, running_loss / len(dataset.train_loader)))
    print('#finish train#')
    return net

def eval(net, device):
    print('#start eval#')
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(dataset.test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print('#finish eval#')

def test(net, device):
    print('#start test#')
    net.eval()
    with torch.no_grad():
        for data in tqdm(dataset.test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            label = labels[0]
            image = images[0]

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)

            pred = predicted[0]

            label.cpu()
            image.cpu()
            pred.cpu()

            print("Ground truth: ", label_list[label])
            print("Predicted:", label_list[pred])

            #反归一化
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)

            dmean = [-mean/std for mean, std in zip(mean, std)]
            dstd = [1/std for std in std]

            image = transforms.Normalize(dmean, dstd)(image)
            image = transforms.ToPILImage(mode='RGB')(image)

            image.save('test.png')

            break

    print('#end test#')
    
#-----------------------------

if __name__ == '__main__':

    #得到网络实例
    net = resnet.resnet34(100, True)
    #定义损失函数
    criterion = nn.CrossEntropyLoss()
    #定义优化器
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
    #定义设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #定义训练次数
    train_epoch = 1
    
    net.to(device)

    net = train(net, train_epoch, device)

    eval(net, device)

    test(net, device)



