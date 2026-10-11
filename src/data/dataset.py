"""
PyTorch Dataset class for APTOS 2019 Blindness Detection dataset.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class APTOSDataset(Dataset):
    """
    PyTorch Dataset for APTOS 2019 Diabetic Retinopathy classification.

    Args:
        csv_path (str): Path to the split CSV file (containing 'id_code' and 'diagnosis').
        img_dir (str): Directory path containing raw images.
        transform (callable, optional): Optional torchvision transform to be applied on a sample.
    """

    NUM_CLASSES = 5

    def __init__(self, csv_path: str, img_dir: str, transform=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Validate required columns
        if "id_code" not in self.df.columns or "diagnosis" not in self.df.columns:
            raise KeyError("CSV file must contain 'id_code' and 'diagnosis' columns.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = row["id_code"]
        label = int(row["diagnosis"])

        img_name = f"{image_id}.png"
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image file '{img_name}' not found in directory: {self.img_dir}"
            )

        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    @property
    def num_classes(self) -> int:
        return self.NUM_CLASSES
