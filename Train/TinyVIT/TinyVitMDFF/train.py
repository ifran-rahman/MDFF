"""TinyViT-5M - MDFF

Frozen pretrained backbone -> multiscale deep feature fusion (2 scales, depth concatenation) -> IncrementalPCA (20 components) -> regressors.

Run from anywhere:
    python Train/TinyVIT/TinyVitMDFF/train.py --dataset tiny
Outputs (Features/, models/) are written next to this script.
"""
import argparse
import json
import os
import sys
import time

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../..'))
sys.path.insert(0, REPO_ROOT)

from Utils.dataset import SolarRadiance, train_test_split
from Utils.device import set_device
from Utils.evaluate import (train_decision_tree_model, train_lr_model, train_poly_model,
                            train_random_forest_model, train_svm_model)
from Utils.feature_extraction import extract_features
from Utils.process_features import process_features

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='full_dataset', help="key under 'dataset' in config.json")
parser.add_argument('--config', default=os.path.join(REPO_ROOT, 'config.json'))
args = parser.parse_args()

start = time.time()

with open(args.config) as f:
    config = json.load(f)
data_root = os.path.join(REPO_ROOT, config["data_root"])
csv_path = os.path.join(REPO_ROOT, config["dataset"][args.dataset])

os.chdir(SCRIPT_DIR)  # Features/ and models/ are written here

device = set_device()
print('Device:', device)

batch_size = 128
random_state = 42
torch.manual_seed(random_state)

train_data, test_data = train_test_split(csv_path)

import timm

# features_only model (no classification head); the last stage (index 3) is the final conv map
model = timm.create_model('tiny_vit_5m_224.dist_in22k_ft_in1k', pretrained=True, features_only=True).to(device)
model.eval()


def get_feature(x):
    return model(x)[3]

mean = config["dataset"]["mean"]
std = config["dataset"]["std"]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

num_scales = 2  # original 60x80 + one bilinear 2x upscale (120x160)


def extract_train_features(data):
    dataset = SolarRadiance(root_dir=data_root, labels=data, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return extract_features(get_feature, train_loader, device, num_scales=num_scales)


def extract_val_features(data):
    dataset = SolarRadiance(root_dir=data_root, labels=data, transform=transform)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return extract_features(get_feature, val_loader, device, num_scales=num_scales)


model_save_name = 'mdff_tinyvit'

# Extract features and reduce dimension using PCA
train_reduced_features, val_reduced_features = process_features(
    train_data=train_data,
    test_data=test_data,
    feature_dir='Features/',
    batch_size=batch_size,
    n_components=20,
    model_save_name=model_save_name,
    extract_train_fn=extract_train_features,
    extract_val_fn=extract_val_features,
)
print('Train features:', train_reduced_features.shape, 'Val features:', val_reduced_features.shape)

# Train the models
train_lr_model(train_reduced_features, val_reduced_features)
train_decision_tree_model(train_reduced_features, val_reduced_features)
train_random_forest_model(train_reduced_features, val_reduced_features)
train_svm_model(train_reduced_features, val_reduced_features)
train_poly_model(train_reduced_features, val_reduced_features, degree=3)

total_time = time.time() - start
print(f"Run Time: {int(total_time // 60)} minutes and {int(total_time % 60)} seconds")
