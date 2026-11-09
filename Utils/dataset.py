import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class SolarRadiance(Dataset):
    """IR sky images paired with pyranometer irradiance (W/m^2).

    `labels` is a DataFrame whose first column is an image path relative to
    `root_dir` (e.g. `2018_09_04/infrared/1536069450IR.png`) and whose second
    column is the irradiance target.
    """

    def __init__(self, root_dir, labels, transform):
        self.root_dir = root_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.labels.iloc[index, 0])
        target = self.labels.iloc[index, 1]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.resize(image, (80, 60))  # 60 x 80 (H x W)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target)


def load_dataset(root_dir):
    """Build a (name, value) table from a raw GIRASOL folder.

    Image names are returned relative to `root_dir`. Note: the shipped CSVs in
    `data/` define the row order used for the train/val split; regenerating
    them with this function may give a different order.
    """
    ds = pd.DataFrame()
    for date in os.listdir(root_dir):
        infrared_folder = os.path.join(root_dir, date, "infrared")
        csv_path = os.path.join(root_dir, date, "pyranometer", f"{date}.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {date}, missing data.")
            continue
        ds_temp = getDs(infrared_folder, csv_path)
        ds_temp['name'] = ds_temp['name'].apply(lambda img: os.path.join(date, 'infrared', img))
        ds = pd.concat([ds, ds_temp], ignore_index=True)
    return ds


def getDs(path, labels):
    pyranometer = pd.read_csv(labels)
    images = os.listdir(path)

    pyranometer.iloc[:, 0] = pyranometer.iloc[:, 0].astype(int).apply(lambda x: str(x) + 'IR.png')
    filtered = pyranometer[pyranometer.iloc[:, 0].isin(images)]
    filtered.columns = ['name', 'value']
    return filtered.drop_duplicates(subset='name')


def train_test_split(csv_path):
    """Chronological 70/30 split (first 70% of rows train, last 30% validation)."""
    data = pd.read_csv(csv_path)
    train_size = int(0.7 * len(data))
    return data[:train_size], data[train_size:]
