import argparse
import json
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pose_format.torch.masked.collator import zero_pad_collator
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from sign_vq.data.normalize import load_pose_header
from sign_vq.dataset import ZipPoseDataset, DirectoryPoseDataset, PackedDataset
from sign_vq.model import PoseFSQAutoEncoder, AutoEncoderLightningWrapper


def parse_args():
    parser = argparse.ArgumentParser()

    # Define your arguments here
    parser.add_argument('--data-path', type=str, help='Path to training dataset')
    parser.add_argument('--wandb-dir', default=None, type=str, help='Path to wandb directory')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--steps', type=int, default=int(1e7), help='Number of training iterations')
    parser.add_argument('--batch-size', type=int, default=240, help='Batch size')
    parser.add_argument('--loss-hand-weight', type=int, default=10, help='Weight for hand reconstruction loss')
    parser.add_argument('--num-layers', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--codebook-size', type=int, default=2 ** 10,
                        choices=[2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 14, 2 ** 16],
                        help='Estimated number of codes in the VQ model')
    parser.add_argument('--num-codebooks', type=int, default=1, help='Number of codebooks')
    parser.add_argument('--max-pose-length', type=int, default=512, help='Maximum pose length in the dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str,
                        default='gpu' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type to use for training',
                        choices=['bfloat16', 'float16', 'float32'])

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
    affected_points = 2 * (hand_points + 1)  # wrist + hand_points
    # We want the loss to be the same scale across different runs, so we change the default weight accordingly
    default_weight = total_points / ((total_points - affected_points) + (affected_points * hand_weight))

    weights = torch.full((total_points, 1), fill_value=default_weight, dtype=torch.float32)
    for hand in ["RIGHT", "LEFT"]:
        # pylint: disable=protected-access
        wrist_index = header._get_point_index(f"{hand}_HAND_LANDMARKS", "WRIST")
        weights[wrist_index: wrist_index + hand_points, :] = hand_weight
        # pylint: disable=protected-access
        body_wrist_index = header._get_point_index("POSE_LANDMARKS", f"{hand}_WRIST")
        weights[body_wrist_index, :] = hand_weight
    return weights


# pylint: disable=too-many-locals
def main():
    args = parse_args()
    torch_dtype = getattr(torch, args.dtype)

    # internal multiplications use the bfloat16 datatype, if a fast matrix multiplication algorithm is available.
    torch.set_float32_matmul_precision("medium")

    auto_encoder = PoseFSQAutoEncoder(codebook_size=args.codebook_size,
                                      num_codebooks=args.num_codebooks,
                                      num_layers=args.num_layers)
    loss_weights = create_loss_weights(hand_weight=args.loss_hand_weight)
    model = AutoEncoderLightningWrapper(auto_encoder, learning_rate=args.lr, loss_weights=loss_weights)

    if args.data_path.endswith(".zip"):
        dataset = ZipPoseDataset(args.data_path, in_memory=True,
                                 dtype=torch_dtype, max_length=args.max_pose_length)
        training_dataset = dataset.slice(10, None)
        validation_dataset = dataset.slice(0, 10)
        shuffle = True  # Shuffle is only slow without in_memory since the zip file is read sequentially
        num_workers = 0  # Reading from multiple workers errors out since the zip file is read sequentially
    else:
        dataset = DirectoryPoseDataset(args.data_path)
        training_dataset = dataset.slice(10, None)
        validation_dataset = dataset.slice(0, 10)
        shuffle = True
        num_workers = os.cpu_count()

    training_iter_dataset = PackedDataset(training_dataset, max_length=args.max_pose_length, shuffle=shuffle)

    train_dataset = DataLoader(training_iter_dataset,
                               batch_size=args.batch_size,
                               num_workers=num_workers,
                               collate_fn=zero_pad_collator)
    validation_dataset = DataLoader(validation_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=zero_pad_collator)

    logger = WandbLogger(project="sign-language-vq",
                         save_dir=args.wandb_dir,
                         log_model=False, offline=args.wandb_dir is None)
    logger.log_hyperparams(auto_encoder.args_dict)

    # Save model arguments to file
    with open(os.path.join(logger.experiment.dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(auto_encoder.args_dict, f, indent=2)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]

    precision = "bf16-mixed" if args.dtype == "bfloat16" else ("16-mixed" if args.dtype == "float16" else None)
    trainer = pl.Trainer(max_steps=args.steps,
                         logger=logger,
                         callbacks=callbacks,
                         val_check_interval=100_000 // args.batch_size,
                         accelerator=args.device,
                         profiler="simple",
                         precision=precision,
                         gradient_clip_val=1,  # Taken from the Llamma 2 paper
                         )

    # TODO: https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html requiring LightningDataModule
    # tuner = Tuner(trainer)
    # # Auto-scale batch size with binary search
    # tuner.scale_batch_size(model, mode="binsearch")

    trainer.fit(model, train_dataloaders=train_dataset, val_dataloaders=validation_dataset)


if __name__ == '__main__':
    main()
