import io
import random
import time
import zipfile
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

from pose_format.torch.masked.tensor import MaskedTensor


def process_pose(pose, max_length: int):
    tensor_data = torch.tensor(pose['data'], dtype=torch.float32)
    tensor_mask = torch.tensor(pose['mask'], dtype=torch.bool)
    tensor_mask = torch.logical_not(tensor_mask)  # numpy and torch have different mask conventions
    tensor = MaskedTensor(tensor=tensor_data, mask=tensor_mask)

    if max_length is not None:
        offset = random.randint(0, len(tensor) - max_length) \
            if len(tensor) > max_length else 0
        tensor = tensor[offset:offset + max_length]
    return tensor


class ZipPoseDataset(Dataset):
    def __init__(self, zip_path: Path, max_length: int = 512):
        self.max_length = max_length

        # pylint: disable=consider-using-with
        self.zip = zipfile.ZipFile(zip_path, 'r')
        self.files = self.zip.namelist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with self.zip.open(self.files[idx]) as file:
            file_content = file.read()  # Read the entire file content

        # Convert the bytes content to a BytesIO object and load with numpy
        pose_file = io.BytesIO(file_content)
        pose = np.load(pose_file)
        return process_pose(pose, self.max_length)

    def __del__(self):
        self.zip.close()


class HuggingfacePoseDataset(Dataset):
    def __init__(self, dataset_path: Path, max_length: int = 512):
        now = time.time()
        from datasets import load_from_disk

        self.dataset = load_from_disk(str(dataset_path))["train"]
        self.max_length = max_length
        print("Loaded huggingface dataset in", time.time() - now)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pose = self.dataset[idx]
        return process_pose(pose, self.max_length)


class DirectoryPoseDataset(Dataset):
    def __init__(self, directory_path: Path, max_length: int = 512):
        self.directory_path = directory_path
        self.files = list(directory_path.glob('*.npz'))
        self.max_length = max_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pose = np.load(self.files[idx])
        return process_pose(pose, self.max_length)


def benchmark_dataloader(dataset, num_workers: int):
    print(f"{num_workers} workers")
    from torch.utils.data import DataLoader
    from pose_format.torch.masked.collator import zero_pad_collator

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                             collate_fn=zero_pad_collator,
                             num_workers=num_workers)
    for _ in tqdm(islice(data_loader, 200)):
        pass


def benchmark():
    # Benchmark
    datasets = [
        # HuggingfacePoseDataset(Path("/scratch/amoryo/poses/huggingface"), max_length=512),
        ZipPoseDataset(Path('/scratch/amoryo/poses/normalized.zip'), max_length=512),
        DirectoryPoseDataset(Path('/scratch/amoryo/poses/normalized'), max_length=512),
    ]

    for dataset in datasets:
        print("Benchmarking", dataset.__class__.__name__)

        print("Benchmarking dataset")
        print(next(iter(dataset)).shape)
        for _ in tqdm(islice(iter(dataset), 500)):
            pass

        print("Benchmarking data loader")
        benchmark_dataloader(dataset, 0)
        benchmark_dataloader(dataset, 1)
        benchmark_dataloader(dataset, 4)
        benchmark_dataloader(dataset, 8)


if __name__ == "__main__":
    benchmark()
