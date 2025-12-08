# Standard Libraries
import os  # Operating system-specific functions
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
import pickle
from torchvision.models import mobilenet_v3_small, mobilenet_v2
from Utils.dataset import *

from time import time

time_start = time()
# Hyperparameters
img_size = 64
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_state = 42
torch.manual_seed(random_state)

root = '/Users/ifranrahmannijhum/Projects/MDFF copy/MDFF-Copy/tiny_ds.csv'

data = pd.read_csv(root)
# data = data[:15000]
data_len = len(data)
train_size = int(0.7 * data_len)
test_size = data_len - train_size

train_data = data[:train_size]
test_data = data[train_size:]
feature_dir = 'Features/'

os.mkdir(feature_dir) if not os.path.exists(feature_dir) else None

train_root  = 'root'
val_root = 'root'
 
from Utils.extract_train import *
from Utils.extract_val import *
from Utils.perform_pca import perform_pca
from Utils.train import train_lr_model, train_decision_tree_model, train_random_forest_model, train_svm_model, train_poly_model

model = mobilenet_v2(pretrained=True).to(device)
model.eval()

# process_train_batch(root, train_data, model)

# process_val_batch(root, test_data, model)
# print('PCA')
# perform_pca()

reducted_train_features = np.load('Features/mdff_mobilenetv3_train.npy')
reducted_val_features = np.load('Features/mdff_mobilenetv3_val.npy')

# train_lr_model(reducted_train_features, reducted_val_features)
# train_decision_tree_model(reducted_train_features, reducted_val_features)
# train_random_forest_model(reducted_train_features, reducted_val_features)
# train_svm_model(reducted_train_features, reducted_val_features)
train_poly_model(reducted_train_features, reducted_val_features, 3)
# train_svm_model(reducted_train_features, reducted_val_features)
# time_end = time()

# print('Dataset Information:')
# print('Train Data:', len(train_data))
# print('Test Data:', len(test_data))
# print('Time taken:', (time_end - time_start)/60, 'minutes')

