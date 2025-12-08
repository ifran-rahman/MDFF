# Standard Libraries
import os  # Operating system-specific functions
import sys
import json
import cv2  # OpenCV library for computer vision tasks
import pandas as pd  # Pandas library for data manipulation and analysis
import numpy as np  # NumPy for numerical operations

# External Libraries
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules, e.g., nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # Functional operations in neural networks
import torch.optim as optim  # Optimization algorithms, e.g., SGD, Adam, etc.
import torchvision.transforms as transforms  # Image transformations for datasets
from torch.optim.lr_scheduler import StepLR  # Learning rate scheduler for optimization
from torch.utils.data import DataLoader  # Utilities for dataset management and mini-batches

from skimage import io  # Image Input/Output operations for scikit-image library
from tqdm import tqdm  # Progress bar for iterations
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
import joblib

import time

start = time.time()
# Locate the configuration file
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config.json'))

# Load the configuration file
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Dynamically determine the base path
base_path = os.path.abspath(config["base_path"])
sys.path.append(base_path)

from Utils.device import set_device
from Utils.dataset import SolarRadiance, train_test_split
from Utils.process_features import process_features
from Utils.evaluate import *

# Set device and configuration
device = set_device()
print('Device: ', device)

batch_size = 128

# Set random state
random_state = 42
torch.manual_seed(random_state)

# Load dataset
root = config["dataset"]["tiny"]
data = pd.read_csv(root)

train_data, test_data = train_test_split(root=root)
feature_dir = 'Features/'
os.makedirs(feature_dir, exist_ok=True)

# redundant paths, will remove later
train_root = 'root'
val_root = 'root'

from torchvision.models import mobilenet_v2
model = mobilenet_v2(pretrained=True).to(device)
model.eval()

import numpy as np
from sklearn.metrics import mean_squared_error

# Define a function to register forward hooks for feature extraction
# Define a function to register forward hooks for feature extraction
def forward_hook_registration(model, total_layers, layers, outputs):

    # Define a function to append the output of each layer to the outputs list
    def append_output(module, input, output):
        outputs.append(output)

    # Create a list to store the hook handles
    hd = [None] * total_layers
    i = 0

    # Register forward hooks for each layer
    for layer_type, layer_indices in layers.items():
        for layer in layer_indices:
            if layer_type == 'conv':
                hd[i] = model.features[layer][0].register_forward_hook(append_output)
                i += 1
            elif layer_type == 'fc':
                hd[i] = model.classifier[layer].register_forward_hook(append_output)
                i += 1
    return hd, outputs

# Define a function to remove the registered hooks
def remove_hooks(hd, total_layers):
    for i in range(total_layers):
        hd[i].remove()

def pad_to_desired_size(original_tensor, tensor_to_pad):

    # Calculate the difference in height and width
    diff_h = original_tensor.size(2) - tensor_to_pad.size(2)  # Height difference
    diff_w = original_tensor.size(3) - tensor_to_pad.size(3)  # Width difference

    # Ensure the difference is non-negative
    if diff_h < 0 or diff_w < 0:
        raise ValueError("tensor_to_pad has larger dimensions than original_tensor.")

    # Calculate padding for height and width
    pad_top = diff_h // 2
    pad_bottom = diff_h - pad_top
    pad_left = diff_w // 2
    pad_right = diff_w - pad_left

    # Calculate mean value for padding
    mean_value = torch.mean(original_tensor).item()

    # Apply padding
    padded_tensor = F.pad(
        tensor_to_pad,
        (pad_left, pad_right, pad_top, pad_bottom),  # (left, right, top, bottom)
        mode="constant",
        value=mean_value,
    )

    return padded_tensor


# Feature extraction 
train_feature_path = feature_dir + 'train/'
val_feature_path = feature_dir + 'val/'
os.makedirs(train_feature_path, exist_ok=True)
os.makedirs(val_feature_path, exist_ok=True)

layers = {'conv': [18], 'fc': [0]}
total_layers = len(layers['conv'])+len(layers['fc'])

def extract_train_features(idx, data, pca_conv=None):
        mean = config["dataset"]["mean"]
        std = config["dataset"]["std"]

        dataset = SolarRadiance(root_dir=train_root, labels=data, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]))
        # Create the train and validation data loaders
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        train_df = []

        with torch.no_grad():
                for image_tensor, targets in train_loader:

                        targets = targets.to(device)

                        # Create the Gaussian pyramid to upscale the image
                        image_list = [image_tensor]

                        for i in range(0, 1):
                                image_tensor = torch.nn.functional.interpolate(image_tensor, scale_factor=2, mode='nearest')
                                image_list.append(image_tensor)

                        conv_features = []
                        final_layer = None
                        for x, input in enumerate(image_list[::-1], 1):
                                input = input.to(device)
                                outputs = []
                                hd, outputs = forward_hook_registration(model, total_layers, layers, outputs)
                                _ = model(input.to(device))
                                remove_hooks(hd, total_layers)
                                conv_features.append(outputs[0])
                                # if x == 3:
                                #         final_layer = outputs[1]

                        # padd the tensors to match size of the original image tensor
                        i1 = conv_features[0]
                        i2 = pad_to_desired_size(conv_features[0], conv_features[1])
                        
                        # i = torch.mean(torch.stack((i1, i2)), dim=0)
                        i = torch.cat((i1, i2), dim=1)
                        # reshape the tensor to match the batch size
                        conv10 = i.view(len(targets), -1)

                        # Concatenate all feature vectors
                        combined_tr_tensor = np.concatenate((conv10.cpu().numpy(), targets.unsqueeze(1).cpu().numpy()), axis=1)

                        # append to the list
                        train_df.append(combined_tr_tensor)

        concated_train_df = np.vstack(train_df)

        np.save(train_feature_path + 'train_features_{}.npy'.format(idx), concated_train_df)
        
        # pca_conv.partial_fit(concated_train_df[:, :-1])

        return pca_conv


def extract_val_features(idx, data):
        mean = config["dataset"]["mean"]
        std = config["dataset"]["std"]

        dataset = SolarRadiance(root_dir= train_root, labels=data, transform = transforms.Compose([
                                                                                transforms.ToTensor(),
                                                                                transforms.Normalize(mean=mean, std=std)  # Normalize using calculated values
                                                                            ]))

        val_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

        val_df = []

        with torch.no_grad():
                for image_tensor, targets in val_loader:

                        targets = targets.to(device)

                        # Create the Gaussian pyramid
                        image_list = [image_tensor]

                        for i in range(0, 1):
                                image_tensor = torch.nn.functional.interpolate(image_tensor, scale_factor=2, mode='nearest')
                                image_list.append(image_tensor)

                        conv_features = []
                        final_layer = None
                        for x, input in enumerate(image_list[::-1], 1):
                                input = input.to(device)
                                outputs = []

                                hd, outputs = forward_hook_registration(model, total_layers, layers, outputs)
                                _ = model(input.to(device))
                                remove_hooks(hd, total_layers)
                                conv_features.append(outputs[0])
                                # taking the final layer output of image  64 x 64 (originaliamge)
                                if x == 3:
                                        final_layer = outputs[1]

                        i1 = conv_features[0]
                        i2 = pad_to_desired_size(conv_features[0], conv_features[1])

                        i = torch.cat((i1, i2), dim=1)
                        # reshape the tensor to match the batch size
                        conv10 = i.view(len(targets), -1)

                        # Concatenate all feature vectors
                        combined_val_tensor = np.concatenate((conv10.cpu().numpy(), targets.unsqueeze(1).cpu().numpy()), axis=1)

                        val_df.append(combined_val_tensor)

                concated_val_df = np.vstack(val_df)

        np.save(val_feature_path+ 'val_features_{}.npy'.format(idx), concated_val_df)

model_save_name = 'conv_mobilenetv2'

# Extract features and reduce dimension using PCA
train_reduced_features, val_reduced_features = process_features(
    train_data=train_data,
    test_data=test_data,
    feature_dir="Features/",
    batch_size=batch_size,
    n_components=20,
    model_save_name=model_save_name,
    extract_train_fn=extract_train_features,
    extract_val_fn=extract_val_features,
)

directories = {
        "scaler_path": os.path.join(feature_dir, "scaler/"),
        "ipca_path": os.path.join(feature_dir, "ipca/"),
        "train_feature_path": os.path.join(feature_dir, "train/"),
        "val_feature_path": os.path.join(feature_dir, "val/"),
        "p_train_feature_path": os.path.join(feature_dir, "p_train/"),
        "p_val_feature_path": os.path.join(feature_dir, "p_val/"),
    }

train_reduced_features = np.load(f'{directories["train_feature_path"]}/{model_save_name}_train.npy')
val_reduced_features = np.load(f'{directories["val_feature_path"]}/{model_save_name}_val.npy')

# Train the model
train_lr_model(train_reduced_features, val_reduced_features)
train_decision_tree_model(train_reduced_features, val_reduced_features)
train_random_forest_model(train_reduced_features, val_reduced_features)
train_svm_model(train_reduced_features, val_reduced_features)
train_poly_model(train_reduced_features, val_reduced_features, degree=3)

end = time.time()

# Calculate total time in seconds
total_time = end - start

# Convert to minutes and seconds
minutes = int(total_time // 60)
seconds = int(total_time % 60)

# Print formatted runtime
print(f"Run Time: {minutes} minutes and {seconds} seconds")