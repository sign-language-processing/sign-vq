import argparse
import itertools
import os
from pathlib import Path
from typing import List
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from tqdm import tqdm

from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std
from sign_vq.model import PoseFSQAutoEncoder
from sign_vq.pose_reconstruction import run_inference
from sign_vq.poses_to_codes import load_model, process_file


def dropout_joints(pose: Pose, dropout_rate: float):
    confidence = pose.body.confidence
    mask = np.random.rand(*confidence.shape) > dropout_rate
    new_confidence = confidence * mask
    body = NumPyPoseBody(fps=pose.body.fps, data=pose.body.data.data, confidence=new_confidence)
    return Pose(header=pose.header, body=body)


def benchmark_single_pose(model: PoseFSQAutoEncoder, original_pose: Pose, reduced_pose: Pose, only_masked=True):
    new_pose = run_inference(model, reduced_pose, only_masked=only_masked)
    new_pose = pre_process_mediapipe(new_pose)
    new_pose = normalize_mean_std(new_pose)
    return ((original_pose.body.data - new_pose.body.data) ** 2).sum()


def benchmark_pose_reconstructions(model: PoseFSQAutoEncoder, poses: List[Pose], steps=100):
    distances = defaultdict(lambda: defaultdict(list))

    for dropout_rate in tqdm(torch.linspace(0, 1, steps)):
        dropout_rate = float(dropout_rate.item())
        for pose in poses:
            reduced_pose = dropout_joints(pose, dropout_rate)

            masked_distance = benchmark_single_pose(model, pose, reduced_pose, only_masked=True)
            distances["masked"][dropout_rate].append(masked_distance)
            unmasked_distance = benchmark_single_pose(model, pose, reduced_pose, only_masked=False)
            distances["unmasked"][dropout_rate].append(unmasked_distance)

    # create a single chart of the sum of the distances for each dropout rate
    for key, values in distances.items():
        plt.plot(values.keys(), [sum(v) for v in values.values()], label=key)

    plt.xlabel("Dropout rate")
    plt.ylabel("Sum of squared distances")
    plt.yscale("log")
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Benchmark pose reconstruction')
    parser.add_argument('--model', type=str, help='Path to trained model', default="sign/mediapipe-vq")
    parser.add_argument('--directory', type=str, help='Path to pose files', default="../examples")
    parser.add_argument('--num-files', type=int, default=10, help='Number of files to benchmark')
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        raise FileNotFoundError(f"Directory {args.directory} does not exist")

    model = load_model(args.model)

    pose_files = itertools.islice(Path(args.directory).rglob("*.pose"), args.num_files)
    poses = [process_file(f.open('rb')) for f in pose_files]

    with torch.no_grad():
        benchmark_pose_reconstructions(model, poses)


if __name__ == "__main__":
    main()
