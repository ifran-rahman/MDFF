import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from Utils.feature_extraction import *
import os
from skimage import io  # Image Input/Output operations for scikit-image library
# Custom Scripts
from Utils.dataset import *
from tqdm import tqdm

# create dir for saving extracted features
val_feature_path = 'Features/val/'

if not os.path.exists(val_feature_path):
  os.mkdir(val_feature_path)

random_state = 42
torch.manual_seed(random_state)

def extract_val_features(val_root, model, idx, data):
        dataset = SolarRadiance(root_dir= val_root, labels=data, transform = transforms.Compose([transforms.ToTensor()]))

        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        val_df = []

        layers = {'conv': [18], 'fc': [0]}
        total_layers = len(layers['conv'])+len(layers['fc'])

        with torch.no_grad():
                for image_tensor, targets in tqdm(val_loader, desc="Extracting validation features"):

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


def process_val_batch(val_root, test_data, model):
    num_rows = len(test_data)
    batch_size = 100

    i = 0
    while i < num_rows:
        batch_filtered_pyranometer = test_data[i:i+batch_size]
        print("Processing rows {} to {}".format(i, i+batch_size))
        # Place your code here to process the batch of 500 rows
        extract_val_features(val_root, model, i, batch_filtered_pyranometer)
        i += batch_size

