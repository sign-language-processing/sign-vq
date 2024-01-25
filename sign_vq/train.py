import argparse
import json
import os
import random

import numpy as np
import torch
from pose_format.torch.masked.collator import zero_pad_collator
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from sign_vq.data.normalize import load_pose_header
from sign_vq.dataset import ZipPoseDataset, DirectoryPoseDataset
from sign_vq.model import PoseFSQAutoEncoder, AutoEncoderLightningWrapper


def parse_args():
    parser = argparse.ArgumentParser()

    # Define your arguments here
    parser.add_argument('--data-path', type=str, help='Path to training dataset')
    parser.add_argument('--wandb-dir', type=str, help='Path to wandb directory')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--steps', type=int, default=int(1e6), help='Number of training iterations')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--loss-hand-weight', type=int, default=5, help='Weight for hand reconstruction loss')
    parser.add_argument('--codebook-size',
                        choices=[2 ** 8, 2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16], default=2 ** 10,
                        help='Estimated number of codes in the VQ model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str,
                        default='gpu' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        torch.random.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    return args


def create_loss_weights(hand_weight=1):
    header = load_pose_header()

    total_points = header.total_points()
    hand_points = 21
    # We want the loss to be the same scale across different runs, so we change the default weight accordingly
    default_weight = total_points / ((total_points - 2 * hand_points) + (2 * hand_points * hand_weight))

    weights = torch.full((total_points, 1), fill_value=default_weight, dtype=torch.float32)
    for hand in ["RIGHT", "LEFT"]:
        # pylint: disable=protected-access
        wrist_index = header._get_point_index(f"{hand}_HAND_LANDMARKS", "WRIST")
        weights[wrist_index: wrist_index + hand_points, :] = hand_weight
    return weights


def main():
    args = parse_args()

    # internal multiplications use the bfloat16 datatype, if a fast matrix multiplication algorithm is available.
    torch.set_float32_matmul_precision("medium")

    auto_encoder = PoseFSQAutoEncoder(args.codebook_size)
    loss_weights = create_loss_weights(hand_weight=args.loss_hand_weight)
    model = AutoEncoderLightningWrapper(auto_encoder, learning_rate=args.lr, loss_weights=loss_weights)

    if args.data_path.endswith(".zip"):
        dataset = ZipPoseDataset(args.data_path)
        shuffle = False  # Shuffle is slow since the zip file is read sequentially
        num_workers = 1  # Reading from multiple workers errors out since the zip file is read sequentially
    else:
        dataset = DirectoryPoseDataset(args.data_path)
        shuffle = True
        num_workers = os.cpu_count()

    train_dataset = DataLoader(dataset,
                               batch_size=args.batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               collate_fn=zero_pad_collator)
    validation_dataset = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=zero_pad_collator)

    logger = WandbLogger(project="sign-language-vq",
                         save_dir=args.wandb_dir,
                         log_model=False, offline=False)

    # Save model arguments to file
    with open(os.path.join(logger.experiment.dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(auto_encoder.args_dict, f, indent=2)

    # callbacks = [ModelCheckpoint(
    #     dirpath="checkpoints/" + logger.experiment.name,
    #     filename="model",
    #     verbose=True,
    #     save_top_k=1,
    #     monitor='validation_loss',
    #     mode='min'
    # )]
    callbacks = []

    trainer = pl.Trainer(max_steps=args.steps,
                         logger=logger,
                         callbacks=callbacks,
                         val_check_interval=5000,
                         limit_val_batches=1,
                         accelerator=args.device,
                         profiler="simple")

    trainer.fit(model, train_dataloaders=train_dataset, val_dataloaders=validation_dataset)


if __name__ == '__main__':
    main()
