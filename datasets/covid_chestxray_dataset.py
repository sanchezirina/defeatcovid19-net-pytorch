import numpy as np
import pandas as pd
from PIL import Image

import cv2
from torch.utils.data import Dataset


class COVIDChestXRayDataset(Dataset):
    def __init__(self, path, size=128, augment=None):
        super(COVIDChestXRayDataset, self).__init__()
        self.path = path
        self.size = size
        self.augment = augment
        self.labels = None
        self.df = None

    def build(self):
        print(f"{self.__class__.__name__} initialized with size={self.size}, augment={self.augment}")
        print(f"Dataset is located in {self.path}")

        image_dir = self.path / "images"
        metadata_path = self.path / "metadata.csv"

        df_metadata = pd.read_csv(metadata_path, header=0)
        # Drop CT scans
        df_metadata = df_metadata[df_metadata["modality"] == "X-ray"]
        # Keep only PA/AP/AP Supine, drop Axial, L (lateral)
        allowed_views = ["PA", "AP", "AP Supine"]
        df_metadata = df_metadata[df_metadata["view"].isin(allowed_views)]

        # COVID-19 = 1, SARS/ARDS/Pneumocystis/Streptococcus/No finding = 0
        self.labels = (df_metadata.finding == "COVID-19").values.reshape(-1, 1)
        images = df_metadata.filename
        images = images.apply(lambda x: image_dir / x).values.reshape(-1, 1)

        self.df = pd.DataFrame(np.concatenate((images, self.labels), axis=1), columns=["image", "label"])

        del images

        print(f"Dataset: {self.df}")

    @staticmethod
    def _load_image(path, size):
        img = Image.open(path)
        img = cv2.resize(np.array(img), (size, size), interpolation=cv2.INTER_AREA)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
            img = np.dstack([img, img, img])
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # size, size, chan -> chan, size, size
        img = np.transpose(img, axes=[2, 0, 1])

        return img

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = self._load_image(row["image"], self.size)
        label = row["label"]

        if self.augment is not None:
            img = self.augment(img)

        return img, label

    def __len__(self):
        return self.df.shape[0]
