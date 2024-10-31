import io
import random
import time
import zipfile
from itertools import islice
from pathlib import Path

import numpy as np
import psutil
import torch
from pose_format.torch.masked import MaskedTorch
from pose_format.torch.masked.tensor import MaskedTensor
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm


def print_memory():
    # Get current process
    process = psutil.Process()

    # Get the memory info of the current process
    memory_info = process.memory_info()

    # Convert bytes to GB
    rss_in_gb = memory_info.rss / (1024 ** 3)
    vms_in_gb = memory_info.vms / (1024 ** 3)

    # Print the RSS and VMS in GB
    print(f"Memory used in GB: RSS={rss_in_gb:.2f}, VMS={vms_in_gb:.2f}")


def preprocess_pose(pose, dtype=torch.float16):
    tensor_data = torch.tensor(pose['data'], dtype=dtype)
    tensor_mask = torch.tensor(pose['mask'], dtype=torch.bool)
    tensor_mask = torch.logical_not(tensor_mask)  # numpy and torch have different mask conventions
    tensor = MaskedTensor(tensor=tensor_data, mask=tensor_mask)

    return tensor


def crop_pose(tensor, max_length: int):
    if max_length is not None:
        offset = random.randint(0, len(tensor) - max_length) \
            if len(tensor) > max_length else 0
        return tensor[offset:offset + max_length]
    return tensor


class _ZipPoseDataset(Dataset):
    def __init__(self, zip_obj: zipfile.ZipFile,
                 files: list,
                 max_length: int = 512,
                 in_memory: bool = False,
                 dtype=torch.float16):
        self.max_length = max_length
        self.zip = zip_obj
        self.files = files
        self.in_memory = in_memory
        self.dtype = dtype
        self.memory_files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if len(self.memory_files) == len(self.files):
            tensor = self.memory_files[idx]
        else:
            # If we want to store in memory, we first load sequentially all the files
            idx = idx if not self.in_memory else len(self.memory_files)

            with self.zip.open(self.files[idx]) as file:
                file_content = file.read()  # Read the entire file content

            # Convert the bytes content to a BytesIO object and load with numpy
            pose_file = io.BytesIO(file_content)
            pose = np.load(pose_file)
            tensor = preprocess_pose(pose, dtype=self.dtype)
            if self.in_memory:
                self.memory_files.append(tensor)
                if len(self.memory_files) % 10000 == 0:
                    print_memory()

        cropped_pose = crop_pose(tensor, self.max_length)
        if cropped_pose.dtype != self.dtype:
            cropped_pose = MaskedTensor(tensor=cropped_pose.tensor.type(self.dtype),
                                        mask=cropped_pose.mask)
        return cropped_pose

    def slice(self, start, end):
        return _ZipPoseDataset(zip_obj=self.zip, files=self.files[start:end],
                               max_length=self.max_length, in_memory=self.in_memory, dtype=self.dtype)


class ZipPoseDataset(_ZipPoseDataset):
    def __init__(self, zip_path: Path, max_length: int = 512, in_memory: bool = False, dtype=torch.float32):
        print(f"ZipPoseDataset @ {zip_path} with max_length={max_length}, in_memory={in_memory}")

        # pylint: disable=consider-using-with
        self.zip_obj = zipfile.ZipFile(zip_path, 'r')
        files = self.zip_obj.namelist()
        print("Total files", len(files))

        super().__init__(zip_obj=self.zip_obj, files=files,
                         max_length=max_length, in_memory=in_memory, dtype=dtype)

    def __del__(self):
        self.zip_obj.close()


class PackedDataset(IterableDataset):
    def __init__(self, dataset: Dataset, max_length: int, shuffle=True):
        self.dataset = dataset
        self.max_length = max_length
        self.shuffle = shuffle

    def __iter__(self):
        dataset_len = len(self.dataset)
        datum_idx = 0

        datum_shape = self.dataset[0].shape
        padding_shape = tuple([10] + list(datum_shape)[1:])
        padding = MaskedTensor(tensor=torch.zeros(padding_shape), mask=torch.zeros(padding_shape))

        while True:
            poses = []
            total_length = 0
            while total_length < self.max_length:
                if self.shuffle:
                    datum_idx = random.randint(0, dataset_len - 1)
                else:
                    datum_idx = (datum_idx + 1) % dataset_len

                # Append pose
                pose = self.dataset[datum_idx]
                poses.append(pose)
                total_length += len(pose)

                # Append padding
                poses.append(padding)
                total_length += len(padding)

            concatenated_pose = MaskedTorch.cat(poses, dim=0)[:self.max_length]
            yield concatenated_pose


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
        return crop_pose(preprocess_pose(pose), self.max_length)


class DirectoryPoseDataset(Dataset):
    def __init__(self, directory_path: Path, max_length: int = 512):
        self.directory_path = directory_path
        self.files = list(directory_path.glob('*.npz'))
        self.max_length = max_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pose = np.load(self.files[idx])
        return crop_pose(preprocess_pose(pose), self.max_length)


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
