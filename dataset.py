import hashlib
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
            "0054e3fc6c06b7a1b436cd26c96714b98fab7388a0f9396c41007822c04b119a"
        )
    ]

    # TODO: CHARS DICT here so we can use it to return list of float in load_data
    corpus = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

    corpus_dict = {char: idx for idx, char in enumerate(corpus)}

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
        target = [self.corpus_dict[char] for char in target]
        target = torch.tensor(target, dtype=torch.long)

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
        ) and all(
            os.path.exists(os.path.join(self.raw_folder, filename.replace(".tar.gz", "")))
            for filename, signature in self.resources
        )

    @staticmethod
    def _check_integrity(filename: str, signature: str):
        if not os.path.isfile(filename):
            return RuntimeError("File not found or corrupted.")

        sha256 = hashlib.sha256()
        with open(filename, "rb") as f:
            while chunk := f.read(1024 * 1024):
                sha256.update(chunk)

        if signature != sha256.hexdigest():
            raise RuntimeError(f"The file signature does not match.")

    def download(self) -> None:
        """Download the data if it does not exist already."""
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        for filename, signature in self.resources:
            for mirror in self.mirrors:
                # url = f"{mirror}/{signature}/{filename}"
                url = f"{mirror}/{filename}"
                try:
                    print(f"Downloading {url}")
                    self._download_and_extract_archive(url,
                                                       download_root=self.raw_folder,
                                                       filename=filename,
                                                       sha256=signature)
                except Exception as e:
                    print(f"Failed to download {url} (trying next):\n{e}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def _download_and_extract_archive(self, url, download_root, filename, sha256) -> None:
        print(f"Downloading {url} to {download_root} as {filename}")

        fpath = f"{download_root}/{filename}"

        import requests
        response = requests.get(url, stream=True)

        size = int(response.headers.get("content-length", 0))

        with tqdm(total=size, unit="bit") as progress:
            with open(fpath, 'wb') as f:
                for data in response.iter_content(1024):
                    f.write(data)
                    progress.update(len(data))
                f.close()

        self._check_integrity(f"{self.raw_folder}/{filename}", sha256)

        print(f"Extracting {filename} to {download_root}")
        try:
            with tarfile.open(fpath, "r:gz") as tar:
                tar.extractall(download_root)
        except Exception as e:
            print(f"Error extracting file: {e}")
