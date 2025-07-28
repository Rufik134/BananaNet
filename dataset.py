from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class BananaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # we map ripeness classes to numeric labels
        self.class_to_idx = {
            'Class A': 0,  # unripe
            'Class B': 1,  # ripe
            'Class C': 2,  # nearly overripe
            'Class D': 3   # overripe
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        image = Image.open(img_path).convert("RGB")

        ripeness_class = self.class_to_idx[row['ripeness_class']]
        ttl_days_left = float(row['ttl_days_left']) / 27.0  #  normalize ttl

        if self.transform:
            image = self.transform(image)

        return image, ripeness_class, ttl_days_left
