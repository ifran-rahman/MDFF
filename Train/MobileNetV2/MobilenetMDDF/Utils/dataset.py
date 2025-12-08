# Standard Libraries
import os  # Operating system-specific functions
import cv2  # OpenCV library for computer vision tasks
import pandas as pd  # Pandas library for data manipulation and analysis

# External Libraries
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules, e.g., nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # Functional operations in neural networks
import torch.optim as optim  # Optimization algorithms, e.g., SGD, Adam, etc.
import torchvision.transforms as transforms  # Image transformations for datasets
from torch.optim.lr_scheduler import StepLR  # Learning rate scheduler for optimization
from torch.utils.data import Dataset, DataLoader  # Utilities for dataset management and mini-batches

from skimage import io  # Image Input/Output operations for scikit-image library
# Custom Scripts
import pickle

# Hyperparameters
img_size = 64
batch_size = 1
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np

class SolarRadiance(Dataset):
    def __init__(self, root_dir, labels, transform):
        self.root_dir = root_dir
        self.labels = labels
        self.transform = transform
        # self.data = self.load_dataset()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.labels.iloc[index,0]
        target = self.labels.iloc[index,1]

        image = io.imread(img_path)
        image = cv2.imread(img_path)#, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (img_size, img_size))

        if self.transform:
             image = self.transform(image)

        y_label = torch.tensor(target)

        return image, y_label


def load_dataset(root_dir):
        ds = pd.DataFrame()
        dates = os.listdir(root_dir)

        try:
            for date in dates:
                infrared_folder = os.path.join(root_dir, date, "infrared")
                pyranometer_folder = os.path.join(root_dir, date, "pyranometer")
                csv_path = os.path.join(pyranometer_folder, "{date}.csv".format(date=date))
                if not os.path.exists(csv_path):
                    print("Skipping date {date} because it does not have both infrared and pyranometer folders".format(date=date))
                    continue
                ds_temp = getDs(infrared_folder, csv_path)
                
                # append the dataframe in the final dataframe
                # ds = ds.append(ds_temp)
                ds = pd.concat([ds, ds_temp], ignore_index=True)

                ds['name'] = ds['name'].apply(lambda img: os.path.join(root_dir, date, 'infrared', img))
        except Exception as e:
            print(e)
        return ds

def getDs(path, labels):
    pyranometer = pd.read_csv(labels)
    images = os.listdir(path)
    
   
    #convert column 1 to int
    X = pyranometer.iloc[:,0].astype(int)

    #convert to image names
    pyranometer.iloc[:,0] = X.apply(lambda x: str(x) + 'IR.png')
    
    # Filter pyranometer DataFrame based on the 'x' column

    filtered_pyranometer = pyranometer[pyranometer.iloc[:,0].isin(images)]
    # Display the result
    filtered_pyranometer.columns = ['name', 'value']

    filtered_pyranometer = filtered_pyranometer.drop_duplicates(subset='name')
    return filtered_pyranometer
