import os
from torchvision import datasets, transforms

# config
# The directory to DLCV_hw1
file_Path = "D:\\Deeplearning\\DLCV_hw1\\"
# The relative directory to training dataset
train_Path = "data\\train"
# The relative directory to validation dataset
val_Path = "data\\val"
input_size = 224
mean = [0.5, 0.5, 0.5, ]
std = [0.5, 0.5, 0.5, ]

# transforms
Transform = {
    'train': transforms.Compose([transforms.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
                                 transforms.RandomHorizontalFlip(0.5),
                                 transforms.RandomRotation(0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std),
                                 ]),
    'validation': transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std),
                                      ])
}

train_Path = os.path.join(file_Path, train_Path)
val_Path = os.path.join(file_Path, val_Path)

train_Datasets = datasets.ImageFolder(root=train_Path, transform=Transform['train'])
val_Datasets = datasets.ImageFolder(root=val_Path, transform=Transform['validation'])
