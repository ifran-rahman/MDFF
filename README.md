# Multiscale Deep Feature Fusion (MDFF) for Solar Estimation from Infrared Sky Images

This repository contains the official implementation of our paper:

**Multiscale Deep Feature Fusion for Solar Estimation from Infrared Sky Images**  
_33rd International Conference on Artificial Intelligence and Cognitive Science (AICS 2025)_  
December 01–02, Hyatt Centric Liberties Hotel, Dublin, Ireland

---

## 📌 Overview

Ground-based infrared (IR) sky images are highly useful for solar irradiance estimation but typically suffer from **low resolution**, **low contrast**, and **weak structural cues**, making traditional feature extraction ineffective.

This work introduces a **Super Resolution–based Multiscale Deep Feature Fusion (MDFF)** pipeline that:

- Generates multiple upscaled versions of the IR image  
- Extracts deep features from a pretrained CNN (MobileNetV2 / EfficientNet / TinyViT)  
- Fuses multiscale features via **depth concatenation**  
- Applies **PCA** for efficient dimensionality reduction  
- Trains lightweight regressors (Linear Regression, Random Forest, Polynomial Regression)

Our experiments show that:
- **MobileNetV2 + Random Forest + Polynomial Regression (degree 3)** provides the best performance  
- Achieves **R² ≈ 0.90**, outperforming single-scale or logits-only baselines

---

## 🚀 Features

- Multiscale deep feature extraction  
- MobileNetV2 backbone (pretrained)  
- IR image super-resolution via bilinear upscaling  
- PCA-based feature compression  
- Multiple regression models:  
  - Linear Regression  
  - Random Forest Regressor  
  - Support Vector Regression  
  - Polynomial Regression  
- Automatic feature saving and loading  
- Evaluation metrics: MSE, RMSE, MAE, R²

---

## 📁 Project Structure

```
MDFF-Solar-Estimation/
│── config.json
│── main_script.py
│── README.md
│── Utils/
│   ├── device.py
│   ├── dataset.py
│   ├── process_features.py
│   └── evaluate.py
│── Features/
│   ├── train/
│   ├── val/
│   ├── scaler/
│   ├── ipca/
│   ├── p_train/
│   └── p_val/
└── models/
```

---

## 🛠 Requirements

Install dependencies:

```
pip install -r requirements.txt
```

If a requirements file is not included:

```
pip install torch torchvision scikit-learn pandas numpy opencv-python tqdm
```

---

## ⚙️ Setup

1. Update `config.json` with:
   - `base_path`
   - dataset path (CSV or directory)
   - normalization mean and std

2. Ensure the following folders exist:

```
Utils/
Features/
config.json
main_script.py
```

---

## ▶️ How to Run

Run the full MDFF pipeline:

```
python main_script.py
```

This will:

- Load the IR dataset  
- Extract multiscale CNN features  
- Apply PCA  
- Train all regression models  
- Save the results and models in the `Features/` directory  

---

## 📤 Outputs

After running, the script will generate:

- Extracted features:
  - `Features/train/*.npy`
  - `Features/val/*.npy`
- PCA-reduced feature sets
- Trained machine-learning models (`.pkl`)
- Evaluation metrics for each model
- Total runtime printed in the console

---

## 📝 Citation

If you use this work, please cite our paper:

```
@inproceedings{MDFF2025,
  title={Multiscale Deep Feature Fusion for Solar Estimation from Infrared Sky Images},
  booktitle={33rd International Conference on Artificial Intelligence and Cognitive Science (AICS)},
  year={2025},
  address={Dublin, Ireland}
}
```

---

## 🔗 Source Code

The source code for the MDFF pipeline is available at:

https://github.com/xxxxxxxx/MDFF-Solar-Estimation  
*(URL censored in the manuscript for anonymity)*

---

## ✉️ Contact

For questions, discussions, or collaboration, feel free to reach out:

**Ifran Rahman Nijhum**  
University College Dublin (UCD)  
ADAPT Centre — THEIA Vision & Analytics Lab  
