import os
import zipfile

import numpy as np
import torch

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Optional, Callable


class CVLicensePlateDataset(Dataset):
    """ Dataset of Commercial Vehicle License Plate."""
    mirrors = [
        "https://cvlpr-dataset.risangbaskoro.com"
    ]

    resources = [
        "cvlpr_cropped_train_v1.zip"
    ]

    # TODO: CHARS DICT here so we can use it to return list of float in load_model
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False
                 ) -> None:
        """ Initialize the dataset.

        Args:
            root: Root path of the dataset.
            train: Whether to get the training, testing, or validation sets.
            transform: Optional transform to be applied on a data sample.
            target_transform: Optional transform to be applied on target sample.
            download: Whether to download the data. If True, downloads the dataset from the internet and puts it in root directory.
        """
        super(CVLicensePlateDataset, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self.load_data()

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def __len__(self):
        """
        Returns:
            Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """ Get the data sample.

        Args:
            idx: Index of the data sample.

        Returns:
            Tuple of the data sample tensor and target tensor.
        """
        img, target = self.data[idx], self.targets[idx]
        # img = Image.fromarray(img.permute((1, 2, 0)).numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def load_data(self):
        images_folder = f"cvlpr_cropped_{'train' if self.train else 'test'}_v1"
        images_path = os.path.join(self.raw_folder, images_folder)
        images = []
        labels = []
        for filename in os.listdir(images_path):
            img_path = os.path.join(images_path, filename)
            img = Image.open(img_path)
            img = np.array(img)
            img = np.moveaxis(img, -1, 0)
            images.append(img)
            label = filename.split("_")[0]
            labels.append(label)

        return torch.tensor(np.array(images)), np.array(labels)

    def _check_exists(self):
        return all(
            os.path.exists(os.path.join(self.root, self.__class__.__name__, "raw", resource))
            for resource in self.resources
        )

    def download(self) -> None:
        """Download the data if it does not exist already."""
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        for filename in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}/{filename}"
                try:
                    print(f"Downloading {url}")
                    self._download_and_extract_archive(url, download_root=self.raw_folder, filename=filename)
                except:
                    print(f"Error downloading {url}")
                finally:
                    print()

    @staticmethod
    def _download_and_extract_archive(url, download_root, filename) -> None:
        print(f"Downloading {url} to {download_root}")

        import requests
        response = requests.get(url, stream=True)

        size = int(response.headers.get("content-length", 0))

        with tqdm(total=size, unit="bit") as progress:
            with open(f"{download_root}/{filename}", 'wb') as f:
                for data in response.iter_content(1024):
                    f.write(data)
                    progress.update(len(data))
                f.close()

        print(f"Extracting {filename} to {download_root}")
        with zipfile.ZipFile(f"{download_root}/{filename}", "r") as zip_ref:
            zip_ref.extractall(download_root)
