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
# Custom Scripts
from Utils.dataset import *
from tqdm import tqdm
from Utils.feature_extraction import *

# create dir for saving extracted features
train_feature_path = 'Features/train/'

if not os.path.exists(train_feature_path):
  os.mkdir(train_feature_path)

random_state = 42
torch.manual_seed(random_state)

def extract_train_features(train_root, model, idx, data, pca_conv):
        dataset = SolarRadiance(root_dir= train_root, labels=data, transform = transforms.Compose([transforms.ToTensor()]))

        # Create the train and validation data loaders
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        train_df = []
  
        layers = {'conv': [18], 'fc': [0]}
        total_layers = len(layers['conv'])+len(layers['fc'])

        with torch.no_grad():
                for image_tensor, targets in tqdm(train_loader, desc="Extracting Training Features"):

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
        
        pca_conv.partial_fit(concated_train_df[:, :-1])

        return pca_conv


from sklearn.decomposition import IncrementalPCA
import joblib

def process_train_batch(train_root, train_data, model):
    batch_size = 100
    num_rows = len(train_data)

    ipca_path = 'Features/ipca/'
    if not os.path.exists(ipca_path):
        os.mkdir(ipca_path)

    ipca_conv = IncrementalPCA(n_components=20, batch_size=batch_size)

    i = 0
    while i < num_rows:
        batch_filtered_pyranometer = train_data[i:i+batch_size]
        print("Processing rows {} to {}".format(i, i+batch_size))
        
        if os.path.exists(ipca_path+'ipca_conv{}.pkl'.format(i)):              
            # Load the PCA checkpoints
            ipca_conv = joblib.load(ipca_path+'ipca_conv{}.pkl'.format(i))
                
        # Place your code here to process the batch of 100 rows
        ipca_conv = extract_train_features(train_root, model, i, batch_filtered_pyranometer, ipca_conv)
        
        # Save the PCA checkpoints
        joblib.dump(ipca_conv, ipca_path+'ipca_conv{}.pkl'.format(i))

        i += batch_size

    # Save the final PCA models
    joblib.dump(ipca_conv, ipca_path+'ipca_conv.pkl')
