import torch
import os
import pandas as pd
from torchvision import transforms
from PIL import Image

from model import vgg_based, device
from Datasets import train_Datasets, Transform

# The directory to DLCV_hw1
file_Path = "D:\\Deeplearning\\DLCV_hw1\\"
# The relative directory to the testing model
model_Path = "model\\Epoch_23.pth"
# The relative directory to out sample file
sample_Path = "sample_test\\"
# The relative directory to the location for saving the csv file
csv_Path = "hw1.csv"

class_name = train_Datasets.classes
df = pd.DataFrame(columns=["image", "class"])

model = vgg_based
model.load_state_dict(torch.load(os.path.join(file_Path, model_Path)))
model = model.to(device)

predicted = []
T = Transform['validation']
filenames = os.listdir(sample_Path)
sample_Path = os.path.join(file_Path, sample_Path)

for filename in filenames:
    img = Image.open(os.path.join(sample_Path, filename))
    img = T(img)
    img = torch.reshape(img, (1, 3, 224, 224))
    img = img.to(device)
    output = model(img)
    pred = torch.argmax(output, dim=1).cpu().detach().numpy()
    predicted.extend(pred)

for i, f in zip(range(10), filenames):
    df = df.append({"image": f, "class": class_name[predicted[i]]}, ignore_index=True)

csv_dir = os.path.join(file_Path, csv_Path)
df.to_csv(csv_dir, index=False)
