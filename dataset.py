import os
import tarfile

import numpy as np
import torch

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Optional, Callable


class CVLicensePlateDataset(Dataset):
    """ Dataset of Commercial Vehicle License Plate."""
    __version__ = 'v1'

    mirrors = [
        "https://data.risangbaskoro.com/cvlpr/master"
    ]

    resources = [
        (
            f"cvlpr_cropped_train_{__version__}.tar.gz",
            "f8e977ce81bbd4d3484285026618090cb6cd08af9031a313e7d831353f01315a"
        )
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
        images_folder = f"cvlpr_cropped_{'train' if self.train else 'test'}_{self.__version__}"
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
            os.path.exists(os.path.join(self.root, self.__class__.__name__, "raw", filename))
            for filename, signature in self.resources
        )

    def download(self) -> None:
        """Download the data if it does not exist already."""
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        for filename, signature in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}/{signature}/{filename}"
                try:
                    print(f"Downloading {url}")
                    self._download_and_extract_archive(url, download_root=self.raw_folder, filename=filename)
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
                finally:
                    print()

    @staticmethod
    def _download_and_extract_archive(url, download_root, filename) -> None:
        print(f"Downloading {url} to {download_root} as {filename}")

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
        try:
            with tarfile.open(f"{download_root}/{filename}", "r:gz") as tar:
                tar.extractall(download_root)
        except Exception as e:
            print(f"Error extracting file: {e}")


if __name__ == '__main__':
    ds = CVLicensePlateDataset("data", download=True)
