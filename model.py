import torch.nn
from torch import optim
from torchvision import models
from torchvision.models import VGG19_Weights

from Datasets import train_Datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vgg_based = models.vgg19(weights=VGG19_Weights.DEFAULT)

for parameter in vgg_based.parameters():
    parameter.requires_grad = False

num_features = vgg_based.classifier[6].in_features
features = list(vgg_based.classifier.children())[:-1]
features.extend([torch.nn.Linear(num_features, len(train_Datasets.classes))])
vgg_based.classifier = torch.nn.Sequential(*features)

vgg_based = vgg_based.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg_based.parameters(), lr=1e-3, momentum=0.9)
