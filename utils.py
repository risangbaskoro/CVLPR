import torch
import matplotlib.pyplot as plt

from const import DECODE_DICT, CHARS_DICT


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


def encode_label(label: str, label_dict: dict = DECODE_DICT):
    """ Encode a label into a sequence of integers representing the character.

    Args:
        label: A string representing the label.
        label_dict: A dictionary mapping characters to integers.

    Returns:
        A list of integers representing the character.
    """
    # Replace huruf "O" dengan angka 0.
    label = label.replace("O", "0")
    return [label_dict[char] for char in label]


def decode_label(input: list, label_dict: dict = CHARS_DICT):
    """ Decode a sequence of integers representing the characters of the label
    Args:
        input: A list of integers representing the characters of the label.
        label_dict: A dictionary mapping integers to characters.

    Returns:
        A list of strings representing the characters.
    """
    return [label_dict[key] for key in input]


def pad_sequence(sequence: list, length: int = 9, mode: str = "pre"):
    """ Pad the target sequence to the given length.

    Args:
        sequence: Sequence to pad.
        length: Length of the target sequence.
        mode: Padding mode (either 'pre' or 'post').

    Raises:
        ValueError

    Returns:
        A list of padded target sequence.
    """
    n_to_pad = length - len(sequence)
    if n_to_pad < 0:
        raise ValueError(f"`length` must be greater than `len(sequence)`. Got len(sequence): {len(sequence)}")

    pad = [0 for i in range(n_to_pad)]
    if mode == "post":
        return sequence + pad
    elif mode == "pre":
        return pad + sequence
    else:
        raise ValueError('Padding must be either "pre" or "post".')
