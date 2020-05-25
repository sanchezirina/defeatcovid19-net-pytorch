from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

import qmenta.client as qclient

PROJECT_NAME = "Kaggle: Chest X-Ray Images (Pneumonia)"


class ChestXRayPneumoniaDataset(Dataset):
    def __init__(self, path, size=128, augment=None):
        super(ChestXRayPneumoniaDataset, self).__init__()
        self.path = path
        self.size = size
        self.augment = augment
        self.labels = None
        self.df = None

    def build(self):
        print("{} initialized with size={}, augment={}".format(self.__class__.__name__, self.size, self.augment))
        print("Dataset is located in {}".format(self.path))

        train_dir = self.path / "train"
        val_dir = self.path / "val"
        test_dir = self.path / "test"

        normal_cases = []
        pneumonia_cases = []
        for folder in [train_dir, val_dir, test_dir]:
            normal_cases.extend((folder / "Normal").glob("*.jpeg"))
            pneumonia_cases.extend((folder / "Pneumonia").glob("*.jpeg"))

        self.labels = np.concatenate((np.zeros(len(normal_cases)), np.ones(len(pneumonia_cases)))).reshape(-1, 1)
        images = np.concatenate((normal_cases, pneumonia_cases)).reshape(-1, 1)

        self.df = pd.DataFrame(np.concatenate((images, self.labels), axis=1), columns=["image", "label"])

        del images

        print("Dataset: {}".format(self.df))

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

    def download_from_platform(self, usr, platform_psw, dst_path):
        dst_path = Path(dst_path)

        # Connect to QMENTA platform
        account = qclient.Account(usr, platform_psw)

        project = account.get_project(PROJECT_NAME)
        metadata = project.get_subjects_metadata()

        num_containers = len(metadata)
        total_num_files = 0

        # Go over all subjects in this project
        for count, container in enumerate(metadata, start=1):
            patient_sn = container["patient_secret_name"]
            ssid = container["ssid"]

            # Get container files and metadata, make sure they exist
            container_id = int(container["container_id"])
            container_files = project.list_container_files(str(container_id))
            if not container_files:
                continue
            container_files_metadata = [
                x for x in project.list_container_files_metadata(str(container_id)) if x["name"] in container_files
            ]
            if type(container_files_metadata) == bool:
                print(f"No files in container of session {patient_sn!r} / {ssid!r}")
                continue

            split = container["md_split"]
            split_path = dst_path / split
            group = container["md_group"]
            group_path = split_path / group

            for file_data in container_files_metadata:
                file_name = file_data["name"]
                if not group_path.is_dir():
                    group_path.mkdir(parents=True, exist_ok=True)
                file_path = group_path / file_name
                if file_path.is_file():
                    print(f"{file_name} already exists in {file_path} ")
                else:
                    project.download_file(container_id, file_name, str(file_path), True)
                    print(f"Downloaded {file_name} from {patient_sn}/{ssid}")

                # Count downloaded file
                total_num_files += 1

            if count % 100 == 0:
                print(f"Processed {count}/{num_containers}")
                print(f"Total number of downloaded files: {total_num_files}")
