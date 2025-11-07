#!/usr/bin/env python3
"""Training script for RVQ-VAE on OpenPose motion data."""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from sign_vq.openpose_dataset import OpenPoseMotionDataset, collate_fn
from sign_vq.rvq_model import RVQ_VAE, RVQ_VAE_Lightning


def parse_args():
    parser = argparse.ArgumentParser(description='Train RVQ-VAE on OpenPose motion data')

    # Data
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to OpenPose JSON file')
    parser.add_argument('--max-length', type=int, default=150,
                        help='Maximum sequence length')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize the data')

    # Model
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--num-quantizers', type=int, default=4,
                        help='Number of residual quantizers')
    parser.add_argument('--codebook-size', type=int, default=1024,
                        help='Size of each codebook')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Weights
    parser.add_argument('--reconstruction-weight', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--commitment-weight', type=float, default=0.25,
                        help='Weight for commitment loss')

    # Logging
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--exp-name', type=str, default='rvq_vae_openpose',
                        help='Experiment name')

    # Device
    parser.add_argument('--accelerator', type=str, default='auto',
                        help='Accelerator type (auto, gpu, cpu)')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices')

    # Checkpointing
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed for reproducibility
    pl.seed_everything(42)

    print("=" * 80)
    print("Loading dataset...")
    print("=" * 80)

    # Load dataset
    dataset = OpenPoseMotionDataset(
        json_path=args.data_path,
        max_length=args.max_length,
        normalize=args.normalize,
        dtype=torch.float32
    )

    # Split into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print("=" * 80)
    print("Creating model...")
    print("=" * 80)

    # Get input dimension from dataset
    sample = dataset[0]
    num_keypoints = sample.shape[1]  # Should be 25 for OpenPose
    input_dim = num_keypoints * 3

    print(f"Input shape: {sample.shape}")
    print(f"Input dimension: {input_dim} ({num_keypoints} keypoints × 3)")

    # Create model
    model = RVQ_VAE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_quantizers=args.num_quantizers,
        codebook_size=args.codebook_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    # Create Lightning module
    lightning_model = RVQ_VAE_Lightning(
        model=model,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        reconstruction_weight=args.reconstruction_weight,
        commitment_weight=args.commitment_weight
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("=" * 80)
    print("Setting up training...")
    print("=" * 80)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.log_dir) / args.exp_name / 'checkpoints',
        filename='rvq-vae-{epoch:02d}-{val/loss:.4f}',
        save_top_k=3,
        monitor='val/loss',
        mode='min',
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_name
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
    )

    print("=" * 80)
    print("Starting training...")
    print("=" * 80)

    # Train
    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.checkpoint_path
    )

    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()
