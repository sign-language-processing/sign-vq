import argparse
import os
import random

import numpy as np
import torch
from pose_format.torch.masked.collator import zero_pad_collator
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl

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


def main():
    args = parse_args()

    # internal multiplications use the bfloat16 datatype, if a fast matrix multiplication algorithm is available.
    torch.set_float32_matmul_precision("medium")

    auto_encoder = PoseFSQAutoEncoder(args.codebook_size)
    model = AutoEncoderLightningWrapper(auto_encoder, learning_rate=args.lr)

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
