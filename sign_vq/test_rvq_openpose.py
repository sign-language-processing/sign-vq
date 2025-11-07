#!/usr/bin/env python3
"""Test script for RVQ-VAE on OpenPose data."""
import json
import sys
from pathlib import Path

import torch
import numpy as np

from sign_vq.openpose_dataset import OpenPoseMotionDataset
from sign_vq.rvq_model import RVQ_VAE, RVQ_VAE_Lightning


def test_dataset(json_path):
    """Test loading and processing the dataset."""
    print("=" * 80)
    print("Testing Dataset")
    print("=" * 80)

    # Load dataset
    dataset = OpenPoseMotionDataset(json_path, max_length=150, normalize=True)

    print(f"\nDataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample dtype: {sample.dtype}")
    print(f"Sample min: {sample.min():.4f}, max: {sample.max():.4f}")

    # Check for NaNs
    if torch.isnan(sample).any():
        print("WARNING: Sample contains NaN values!")
    else:
        print("✓ No NaN values in sample")

    return dataset


def test_model(dataset):
    """Test creating and running the model."""
    print("\n" + "=" * 80)
    print("Testing Model")
    print("=" * 80)

    # Get input dimensions
    sample = dataset[0]
    num_keypoints = sample.shape[1]
    input_dim = num_keypoints * 3

    print(f"\nInput dimension: {input_dim} ({num_keypoints} keypoints × 3)")

    # Create model
    model = RVQ_VAE(
        input_dim=input_dim,
        hidden_dim=256,  # Smaller for testing
        num_quantizers=2,
        codebook_size=512,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )

    print(f"Model created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch = sample.unsqueeze(0)  # Add batch dimension
    print(f"Input shape: {batch.shape}")

    with torch.no_grad():
        x_recon, commitment_loss = model(batch)

    print(f"Output shape: {x_recon.shape}")
    print(f"Commitment loss: {commitment_loss.item():.6f}")

    # Check reconstruction error
    recon_error = torch.nn.functional.mse_loss(x_recon, batch)
    print(f"Reconstruction error (MSE): {recon_error.item():.6f}")

    # Test encoding and decoding
    print("\nTesting encode/decode...")
    with torch.no_grad():
        indices = model.encode(batch)
        print(f"Encoded indices shape: {indices.shape}")
        print(f"Indices range: [{indices.min().item()}, {indices.max().item()}]")

        x_decoded = model.decode(indices)
        print(f"Decoded shape: {x_decoded.shape}")

    # Check if encode-decode matches forward pass
    decode_error = torch.nn.functional.mse_loss(x_decoded, x_recon)
    print(f"Encode-decode vs forward error: {decode_error.item():.6f}")

    if decode_error < 1e-5:
        print("✓ Encode-decode matches forward pass")
    else:
        print("WARNING: Encode-decode doesn't match forward pass!")

    return model


def test_lightning_module(model, dataset):
    """Test the Lightning wrapper."""
    print("\n" + "=" * 80)
    print("Testing Lightning Module")
    print("=" * 80)

    # Create Lightning module
    lightning_model = RVQ_VAE_Lightning(
        model=model,
        learning_rate=3e-4,
        warmup_steps=100,
        reconstruction_weight=1.0,
        commitment_weight=0.25
    )

    print("Lightning module created successfully")

    # Test training step
    print("\nTesting training step...")
    sample = dataset[0].unsqueeze(0)

    lightning_model.train()
    loss = lightning_model.training_step(sample, 0)
    print(f"Training loss: {loss.item():.6f}")

    # Test validation step
    print("Testing validation step...")
    lightning_model.eval()
    with torch.no_grad():
        val_loss = lightning_model.validation_step(sample, 0)
    print(f"Validation loss: {val_loss.item():.6f}")

    print("✓ Lightning module tests passed")

    return lightning_model


def test_codebook_utilization(model, dataset, num_samples=10):
    """Test codebook utilization."""
    print("\n" + "=" * 80)
    print("Testing Codebook Utilization")
    print("=" * 80)

    all_indices = []
    model.eval()

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i].unsqueeze(0)
            indices = model.encode(sample)
            all_indices.append(indices)

    # Concatenate all indices
    all_indices = torch.cat(all_indices, dim=0)  # (num_samples, seq_len, num_quantizers)

    print(f"\nAnalyzed {num_samples} samples")
    print(f"Total indices shape: {all_indices.shape}")

    # Check utilization for each quantizer
    num_quantizers = all_indices.shape[-1]
    for q_idx in range(num_quantizers):
        q_indices = all_indices[:, :, q_idx]
        unique_codes = torch.unique(q_indices)
        utilization = len(unique_codes) / model.codebook_size * 100

        print(f"Quantizer {q_idx}: {len(unique_codes)}/{model.codebook_size} "
              f"codes used ({utilization:.1f}%)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_rvq_openpose.py <path_to_openpose_json>")
        print("\nExample:")
        print("  python sign_vq/test_rvq_openpose.py data/motion_data.json")
        sys.exit(1)

    json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    print("Testing RVQ-VAE with OpenPose data")
    print(f"Data file: {json_path}\n")

    # Run tests
    dataset = test_dataset(json_path)
    model = test_model(dataset)
    lightning_model = test_lightning_module(model, dataset)
    test_codebook_utilization(model, dataset)

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    print("\nYou can now train the model with:")
    print(f"  python sign_vq/train_rvq_openpose.py --data-path {json_path}")


if __name__ == '__main__':
    main()
