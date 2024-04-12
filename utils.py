import torch
import matplotlib.pyplot as plt


def plot_dataset_images(dataset, rows=5, cols=5, seed=None):
    """ Plot the dataset's images and their corresponding labels.

    Args:
        dataset: A torch.utils.data.Dataset instance. Most likely be `CVLicensePlate`.
        rows: Number of rows to be displayed.
        cols: Number of cols to be displayed.
        seed: Random seed for the random number. If None, no seed will be used.

    Returns:
        None
    """
    if seed is not None:
        torch.manual_seed(seed)
    plt.figure(figsize=(cols * 2, rows))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        idx = torch.randint(len(dataset), (1,)).item()
        image, label = dataset[idx]
        plt.imshow(image.permute(1, 2, 0))
        plt.title(f"{label} ({len(label)})")
        plt.axis(False)
