<p align=center>
  <img src="https://github.com/risangbaskoro/CVLPR/assets/36572342/dcd9ddb6-8b2e-40e2-80db-0479024fdc39)"  alt="License plate images with their true label as title for each image"/>
</p>
<h1 align="center">
    Commercial Vehicle License Plate Recognition
</h1>

The repository for Commercial Vehicle License Plate Recognition (CVLPR), inspired by
[LPRNet](https://arxiv.org/abs/1806.10447).

> [!WARNING]
> This project is **Work In Progress**. This repo is considered ready when this message is no longer exists.

This repository is built mainly with [PyTorch](https://pytorch.org).

## Setup

1. Download this repository:

```shell
git clone https://github.com/risangbaskoro/cvlpr.git
```

2. Install all dependencies:

```shell
pip install -r requirements.txt
```

## Usage

> [!NOTE]
> Work in progress...

### Dataset

The dataset of this project can be found using one of these methods:

- Using the provided `CVLicensePlateDataset` class from `dataset.py` file.
- Download via https://data.risangbaskoro.com/cvlpr/master/<dataset_name.tar.gz>

> [!TIP]
> I would recommend you to download the dataset using the provided class, as that class would be updated everytime
> there's an update to the dataset itself.

## To Do

- [ ] Write a better documentation.
- [ ] Write a script to automatically compress new dataset using `gzip` and append `sha256` of its contents as the
  filename. E.g.: `main/cvlpr/data/<sha256>-cvlpr-dataset-<version>.gz`.
- [ ] With that in mind, also adjust the download resource from `CVLicensePlateDataset`.
- [ ] Implement Global Context as in [ParseNet](https://arxiv.org/abs/1506.04579).
- [ ] Create (and implement) Beam Search for the decoding procedure at inference stage.

## Contact

You can contact me at contact@risangbaskoro.com.