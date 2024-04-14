#!/usr/bin/env python

import hashlib
import os
import tarfile
import shutil

import rclone_python.hash_types
from rclone_python import rclone
from rclone_python.remote_types import RemoteTypes
from tqdm.auto import tqdm


def get_file_sha256(filename, buffer_size: int = 65536):
    """
    Get SHA256 hash of a file.
    Args:
        filename: str
            A path to target file
        buffer_size: int
            The size of the buffer to read the file

    Returns:
        String of the SHA256
    """
    hash = hashlib.sha256()

    with open(filename, "rb") as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            hash.update(data)

    return hash.hexdigest()


def check_dataset_signature(file_path):
    """ Compare the file SHA256 against the directory name

    Args:
        file_path: str
            The filename along with the directory to check

    Returns:
        True if the SHA256 hash of the file matches the directory name
    """
    if len(file_path.split("/") < 2):
        raise ValueError("File path must contain the SHA256 directory name.")
    sig = file_path.split("/")[-2]
    file_sig = get_file_sha256(file_path)

    return file_sig == sig


def compress_dataset(target_filename, source_path, dest_path, buffer_size: int = 65536):
    """ Compress the file and save it to dest_path

    Args:
        target_filename: str
            Target filename
        source_path: str
            Source directory
        dest_path: str
            Destination directory
        buffer_size: int
            The size of the buffer to read the file

    Returns:
        A string of the new file with its directory.
    """
    if not target_filename.endswith("tar.gz"):
        raise RuntimeError('`filename` must end with "tar.gz"')

    os.makedirs(dest_path, exist_ok=True)

    tmp_filename = "tmp-" + target_filename
    tmp_path = os.path.join(dest_path, tmp_filename)

    with tarfile.open(tmp_path, mode='w:gz') as tar:
        arcname = source_path.split("/")[-1]
        tar.add(source_path, arcname=arcname)

    signature = get_file_sha256(tmp_path, buffer_size=buffer_size)
    # new_path = os.path.join(dest_path, signature)
    # new_filename = os.path.join(new_path, target_filename)
    new_filename = os.path.join(dest_path, target_filename)

    # os.makedirs(new_path, exist_ok=True)

    os.rename(tmp_path, new_filename)
    print(f"`{target_filename}` is saved with signature `{signature}` in `{new_filename}`")
    return new_filename, signature


if __name__ == '__main__':
    local_destination = 'data/new'
    local_train_path = os.path.join(local_destination, 'train')
    local_test_path = os.path.join(local_destination, 'test')
    local_val_path = os.path.join(local_destination, 'val')

    remote_name = 'r2-data-bucket'
    bucket_path = 'cvlpr/master'
    # bucket_train_path = remote_name + ":" + os.path.join(bucket_path, 'train')
    # bucket_test_path = remote_name + ":" + os.path.join(bucket_path, 'test')
    # bucket_val_path = remote_name + ":" + os.path.join(bucket_path, 'val')

    if os.path.exists(local_destination):
        shutil.rmtree(local_destination)

    new_version = 'v2'

    train_file, train_sig = compress_dataset(f"cvlpr_cropped_train_{new_version}.tar.gz",
                                             'data/CVLicensePlateDataset/raw/cvlpr_cropped_train_v1',
                                             local_train_path)

    # rclone.copy(local_train_path, f"{remote_name}:{bucket_path}")

    with open("dataset.py", "r") as f:
        strings = f.read()
        f.close()

    import dataset

    ds = dataset.CVLicensePlateDataset

    old_version = ds.__version__
    strings = strings.replace(old_version, "v2")

    old_train_sig = ds.resources[0][-1]
    strings = strings.replace(old_train_sig, train_sig)

    with open("dataset.py", "w") as f:
        f.write(strings)
        f.close()
