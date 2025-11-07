"""Dataset loader for OpenPose JSON format motion data."""
import json
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class OpenPoseMotionDataset(Dataset):
    """Dataset for OpenPose motion data in JSON format.

    Expected JSON structure:
    {
        "frames": [
            {
                "frame_idx": 0,
                "openpose_data": {
                    "keypoints_3d": [[x, y, z], ...],  # 25 keypoints
                    "left_hand_keypoints_3d": null,
                    "right_hand_keypoints_3d": null
                }
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        json_path: Union[str, Path],
        max_length: Optional[int] = 512,
        normalize: bool = True,
        dtype=torch.float32
    ):
        """Initialize the dataset.

        Args:
            json_path: Path to the JSON file
            max_length: Maximum sequence length. If None, no cropping is applied.
            normalize: Whether to normalize the data
            dtype: PyTorch data type for tensors
        """
        self.json_path = Path(json_path)
        self.max_length = max_length
        self.normalize = normalize
        self.dtype = dtype

        # Load JSON data
        print(f"Loading OpenPose data from {self.json_path}...")
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        self.frames = data['frames']
        print(f"Loaded {len(self.frames)} frames")

        # Extract keypoints and convert to numpy array
        self.keypoints = []
        for frame in self.frames:
            kps = frame['openpose_data']['keypoints_3d']
            # kps is a list of 25 keypoints, each with [x, y, z]
            if isinstance(kps[0][0], list):
                # Already in nested list format
                kps_array = np.array(kps[0], dtype=np.float32)
            else:
                kps_array = np.array(kps, dtype=np.float32)
            self.keypoints.append(kps_array)

        # Stack into (time, num_keypoints, 3) array
        self.keypoints = np.stack(self.keypoints, axis=0)
        print(f"Keypoints shape: {self.keypoints.shape}")

        # Compute normalization statistics if needed
        if self.normalize:
            self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute mean and std for normalization."""
        # Compute statistics over all valid (non-zero) keypoints
        valid_mask = ~np.all(self.keypoints == 0, axis=-1, keepdims=True)

        # Mean
        self.mean = np.sum(self.keypoints * valid_mask, axis=(0, 1), keepdims=True) / \
                    np.sum(valid_mask, axis=(0, 1), keepdims=True)

        # Std
        centered = (self.keypoints - self.mean) * valid_mask
        self.std = np.sqrt(np.sum(centered ** 2, axis=(0, 1), keepdims=True) /
                          np.sum(valid_mask, axis=(0, 1), keepdims=True))

        # Avoid division by zero
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

        print(f"Normalization - Mean: {self.mean.squeeze()}, Std: {self.std.squeeze()}")

    def __len__(self):
        """Return total number of frames."""
        return len(self.keypoints)

    def __getitem__(self, idx):
        """Get a single sequence.

        Returns:
            torch.Tensor: Keypoints tensor of shape (seq_len, num_keypoints, 3)
        """
        # Get all frames as a single sequence
        keypoints = self.keypoints.copy()

        # Apply normalization
        if self.normalize:
            keypoints = (keypoints - self.mean) / self.std

        # Convert to tensor
        keypoints_tensor = torch.tensor(keypoints, dtype=self.dtype)

        # Crop to max_length if specified
        if self.max_length is not None and len(keypoints_tensor) > self.max_length:
            start_idx = random.randint(0, len(keypoints_tensor) - self.max_length)
            keypoints_tensor = keypoints_tensor[start_idx:start_idx + self.max_length]

        return keypoints_tensor


class OpenPoseMotionDatasetMultiple(Dataset):
    """Dataset that loads multiple OpenPose JSON files."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        max_length: Optional[int] = 512,
        normalize: bool = True,
        dtype=torch.float32
    ):
        """Initialize the dataset.

        Args:
            data_dir: Directory containing JSON files
            max_length: Maximum sequence length
            normalize: Whether to normalize the data
            dtype: PyTorch data type for tensors
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.normalize = normalize
        self.dtype = dtype

        # Find all JSON files
        self.json_files = sorted(list(self.data_dir.glob("*.json")))
        print(f"Found {len(self.json_files)} JSON files in {self.data_dir}")

        if len(self.json_files) == 0:
            raise ValueError(f"No JSON files found in {self.data_dir}")

        # Load all sequences
        self.sequences = []
        all_keypoints = []

        for json_file in self.json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            frames = data['frames']
            keypoints = []
            for frame in frames:
                kps = frame['openpose_data']['keypoints_3d']
                if isinstance(kps[0][0], list):
                    kps_array = np.array(kps[0], dtype=np.float32)
                else:
                    kps_array = np.array(kps, dtype=np.float32)
                keypoints.append(kps_array)

            keypoints = np.stack(keypoints, axis=0)
            self.sequences.append(keypoints)
            all_keypoints.append(keypoints)

        print(f"Loaded {len(self.sequences)} sequences")

        # Compute normalization statistics across all sequences
        if self.normalize:
            all_keypoints = np.concatenate(all_keypoints, axis=0)
            self._compute_normalization_stats(all_keypoints)

    def _compute_normalization_stats(self, all_keypoints):
        """Compute mean and std for normalization."""
        valid_mask = ~np.all(all_keypoints == 0, axis=-1, keepdims=True)

        self.mean = np.sum(all_keypoints * valid_mask, axis=(0, 1), keepdims=True) / \
                    np.sum(valid_mask, axis=(0, 1), keepdims=True)

        centered = (all_keypoints - self.mean) * valid_mask
        self.std = np.sqrt(np.sum(centered ** 2, axis=(0, 1), keepdims=True) /
                          np.sum(valid_mask, axis=(0, 1), keepdims=True))

        self.std = np.where(self.std < 1e-6, 1.0, self.std)

        print(f"Normalization - Mean: {self.mean.squeeze()}, Std: {self.std.squeeze()}")

    def __len__(self):
        """Return number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get a sequence.

        Returns:
            torch.Tensor: Keypoints tensor of shape (seq_len, num_keypoints, 3)
        """
        keypoints = self.sequences[idx].copy()

        if self.normalize:
            keypoints = (keypoints - self.mean) / self.std

        keypoints_tensor = torch.tensor(keypoints, dtype=self.dtype)

        # Crop to max_length if specified
        if self.max_length is not None and len(keypoints_tensor) > self.max_length:
            start_idx = random.randint(0, len(keypoints_tensor) - self.max_length)
            keypoints_tensor = keypoints_tensor[start_idx:start_idx + self.max_length]

        return keypoints_tensor


def collate_fn(batch):
    """Collate function to pad sequences to the same length.

    Args:
        batch: List of tensors of shape (seq_len, num_keypoints, 3)

    Returns:
        Padded tensor of shape (batch, max_seq_len, num_keypoints, 3)
    """
    max_len = max(len(seq) for seq in batch)

    padded_batch = []
    for seq in batch:
        if len(seq) < max_len:
            padding = torch.zeros(max_len - len(seq), *seq.shape[1:], dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_batch.append(padded_seq)

    return torch.stack(padded_batch, dim=0)
