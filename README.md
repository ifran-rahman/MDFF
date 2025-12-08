```markdown
# README

## Requirements
Install dependencies:

pip install -r requirements.txt

If no requirements file is available, install manually:

pip install torch torchvision scikit-learn pandas numpy opencv-python tqdm

## Setup
1. Ensure `config.json` includes:
   - base_path
   - dataset paths
   - normalization mean/std

2. Confirm the project contains:

Utils/  
Features/  
config.json  
main_script.py

## How to Run
Run the main script:

python main_script.py

This will:
- load the dataset
- extract CNN features
- apply PCA
- train regression models
- save outputs in `Features/`

## Outputs
The script produces:
- feature files (`Features/train/*.npy`, `Features/val/*.npy`)
- PCA-reduced features
- trained models (`.pkl`)
- printed evaluation metrics
- total runtime
```
