import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pickle

mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)

transform=transforms.Compose([        
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std)
                ])

train_dataset=datasets.CIFAR100(
                    root='dataset',  
                    train=True,     
                    download=True,  
                    transform=transform
                )
train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=16,  
                    shuffle=True
                )

#---------------------------------------------

test_dataset=datasets.CIFAR100(
                    root='dataset',  
                    train=False,     
                    download=True,  
                    transform=transform
                )
test_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=4,  
                    shuffle=False
                )

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='latin1')
    return dict

dict = unpickle('dataset/cifar-100-python/meta')
# print(dict)

label_list = dict['fine_label_names']