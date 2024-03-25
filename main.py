from torch.utils.data import DataLoader
from Datasets import train_Datasets, val_Datasets
from model import vgg_based, criterion, optimizer
from Train import train

# config
epoch = 25
batch = 10
val_freq = 1
# The relative directory for saving models
save_dir = 'model'
Train_Datasets = train_Datasets
Val_Datasets = val_Datasets

# Dataloaders
train_dataloaders = DataLoader(dataset=Train_Datasets, batch_size=batch, shuffle=True)
val_dataloaders = DataLoader(dataset=Val_Datasets, batch_size=batch, shuffle=False)

vgg_based = train(vgg_based, train_dataloaders, val_dataloaders, criterion, optimizer, epoch, val_freq, save_dir)
