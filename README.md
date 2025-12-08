README
Requirements

Install dependencies:

pip install -r requirements.txt


If no requirements file is provided, install manually:

pip install torch torchvision scikit-learn pandas numpy opencv-python tqdm

Setup

Make sure config.json contains:

base_path

dataset paths

normalization mean/std

Ensure the directory structure includes:

Utils/
Features/
config.json
main_script.py

How to Run

Run the main script:

python main_script.py


This will:

load the dataset

extract CNN features

apply PCA

train regression models

save outputs in Features/

Outputs

The script generates:

extracted feature files (Features/train/*.npy, Features/val/*.npy)

PCA-reduced features

trained models (.pkl)

console evaluation metrics

total runtime printed at the end
