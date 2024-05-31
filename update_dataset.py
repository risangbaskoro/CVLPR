#!/usr/bin/env python

import hashlib
import os
import tarfile
import shutil

from rclone_python import rclone


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
        arcname = target_filename.split(".")[0]
        tar.add(source_path, arcname=arcname)

    signature = get_file_sha256(tmp_path, buffer_size=buffer_size)
    # new_path = os.path.join(dest_path, signature)
    # new_filename = os.path.join(new_path, target_filename)
    new_filename = os.path.join(dest_path, target_filename)

    # os.makedirs(new_path, exist_ok=True)

    os.rename(tmp_path, new_filename)
    print(f"`{target_filename}` is saved with signature `{signature}` in `{new_filename}`")
    return new_filename, signature


def compress_and_upload_dataset(version, source='cropped', dest="data/new", upload=False, commit=False):
    sets = ['train', 'test', 'val']
    local_destination = dest

    local_subset_paths = []
    for subset in sets:
        local_subset_paths.append(os.path.join(local_destination, subset))

    if os.path.exists(local_destination):
        shutil.rmtree(local_destination)

    files = []

    for idx, local_subset_path in enumerate(local_subset_paths):
        file, signature = compress_dataset(f"cvlpr_cropped_{sets[idx]}_{version}.tar.gz",
                                           source,
                                           local_subset_path)

        files.append((file, signature))

    if upload:
        remote_name = 'r2-data-bucket'
        bucket_path = 'cvlpr/master'

        for local_subset_path in local_subset_paths:
            rclone.copy(local_subset_path, f"{remote_name}:{bucket_path}")
            print(f"{local_subset_path} uploaded to {remote_name}:{bucket_path}")

        with open("dataset.py", "r") as f:
            strings = f.read()
            f.close()

        import dataset

        ds = dataset.CVLicensePlateDataset

        old_version = ds.__version__
        strings = strings.replace(old_version, version)

        for idx, (old_file, old_signature) in enumerate(ds.resources):
            strings = strings.replace(old_signature, files[idx][-1])

        with open("dataset.py", "w") as f:
            f.write(strings)
            f.close()

    if commit:
        os.system("git add dataset.py")
        os.system(f'git commit -m "Update dataset.py to version \'{version}\'"')


if __name__ == '__main__':
    import argparse
    from datetime import datetime

    default_version = datetime.now().strftime("%Y%m%d")

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--version", type=str, default=default_version, help="Version of the new dataset.")
    parser.add_argument("-s", "--source", type=str, default="cropped", help="Source of the new dataset.")
    parser.add_argument("-d", "--destination", type=str, default="data/new",
                        help="Destination of the compressed dataset.")
    parser.add_argument("-u", "--upload", type=bool, action=argparse.BooleanOptionalAction,
                        help="Set to upload the compressed dataset to the bucket remote.")
    parser.add_argument("-c", "--commit", type=bool, action=argparse.BooleanOptionalAction,
                        help="Set to automatically commit dataset.py")

    args = parser.parse_args()

    compress_and_upload_dataset(
        version=args.version,
        source=args.source,
        dest=args.destination,
        upload=args.upload,
        commit=args.commit
    )
