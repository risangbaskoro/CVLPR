#!/usr/bin/env python

import argparse
import os

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from LPRNet import LPRNet
from dataset import CVLicensePlateDataset


@dataclass
class TrainConfig:
    """Configuration for training."""
    device: torch.device = torch.device("cpu")
    epochs: int = 1
    lr: float = 1e-2
    batch_size: int = 32
    save_path: Union[str, None] = None
    load_path: Union[str, None] = None

    dict = asdict


def get_arguments():
    parser = argparse.ArgumentParser()
    # TODO: Add arguments here...
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs.')
    parser.add_argument('-r', '--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('-s', '--save_path', type=str, default=None, help='Path to save.')
    parser.add_argument('-l', '--load_path', type=str, default=None, help='Path to load the checkpoint.')
    return parser.parse_args()


def print_arguments(config, num_dashes=30):
    print("Training LPRNet with the following configuration:")
    print("-" * num_dashes)
    for key, value in vars(config).items():
        print(f"{key:<15s}\t: {value}")
    print("-" * num_dashes)


def collate_fn(batch):
    """Collate function for the dataloader.

    Automatically adds padding to the target of each batch.
    """
    # Extract samples and targets from the batch
    samples, targets = zip(*batch)

    # Pad the target sequences to the same length
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    # Return padded samples and targets
    return torch.stack(samples), padded_targets


def train(model, dataloader, config, epoch, start_epoch, optimizer, criterion):
    model.train()
    train_loss = 0

    for idx, batch in enumerate(
            tqdm(dataloader,
                 unit="steps",
                 desc=f"Epoch {epoch + 1}/{start_epoch + config.epochs}",
                 dynamic_ncols=True)
    ):
        # Zero grad optimizer
        optimizer.zero_grad()

        # Define input and target from the batch
        images, targets = batch
        images = images.type(torch.float).to(config.device)
        targets = targets.type(torch.float).to(config.device)

        # Get logits
        logits = model(images)

        # Prepare logits to calculate loss
        logits = logits.mean(dim=2)

        # Calculate each sequence length for each sample
        sample_batch_size, sequence_length = logits.size(0), logits.size(1)
        input_lengths = torch.full(size=(sample_batch_size,), fill_value=sequence_length, dtype=torch.long)

        # Calculate target length for each target sample
        target_lengths = targets.ne(0).sum(dim=1)

        # Transpose the logits
        logits = logits.permute(2, 0, 1)  # (Timestep, Batch Size, Number of Classes)

        log_probs = F.log_softmax(logits, dim=-1)

        # Calculate loss
        loss = criterion(log_probs=log_probs,
                         targets=targets,
                         input_lengths=input_lengths,
                         target_lengths=target_lengths)

        # Backprop
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Add loss
        train_loss += loss

    return train_loss


def save_checkpoint(model, config, epoch, file_prefix="cvlpr_ckpt"):
    version = datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join(config.save_path, version)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(
        save_dir,
        f"{file_prefix}_{epoch + 1}.pth"
    )

    torch.save(model.state_dict(), save_path)
    print(f"Saved to {save_path}")


def get_latest_checkpoint(path):
    ret = path
    if os.path.isdir(path):
        file_epochs = {
            int(file.split('_')[-1].split('.')[0]): file
            for file in os.listdir(path)
        }
        ret = file_epochs[max(file_epochs)]

    to_load = str(os.path.join(path, ret))
    print(f"Loading latest checkpoint from {to_load}")
    return to_load


def main(args):
    config = TrainConfig(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epochs=args.epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        save_path=args.save_path,
        load_path=args.load_path
    )

    print_arguments(config)

    ds = CVLicensePlateDataset("data", download=True)

    train_dl = DataLoader(dataset=ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    num_classes = len(ds.corpus) + 1

    model = LPRNet(num_classes=num_classes,
                   input_channel=3,
                   with_global_context=False)

    start_epoch = 0
    if config.load_path:
        file_path = get_latest_checkpoint(config.load_path)
        start_epoch = int(file_path.split('_')[-1].split('.')[0])
        model.load_state_dict(torch.load(file_path, map_location=config.device))

    model.to(config.device)

    loss_fn = nn.CTCLoss(blank=0, zero_infinity=False, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr)

    for epoch in range(start_epoch, start_epoch + config.epochs):
        train_loss = train(model=model,
                           dataloader=train_dl,
                           config=config,
                           epoch=epoch,
                           start_epoch=start_epoch,
                           optimizer=optimizer,
                           criterion=loss_fn)

        print(f"Loss: {train_loss / train_dl.batch_size}")

        # Save Checkpoint
        if config.save_path and (epoch + 1) % 25 == 0:
            save_checkpoint(
                model=model,
                config=config,
                epoch=epoch)

        print()


if __name__ == '__main__':
    arguments = get_arguments()
    main(arguments)
