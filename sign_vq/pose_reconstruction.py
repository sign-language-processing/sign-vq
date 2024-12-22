import argparse
import os

import torch
from pose_format import Pose

from sign_vq.model import PoseFSQAutoEncoder
from sign_vq.poses_to_codes import load_model, process_file, pose_to_tensor
from sign_vq.utils import pose_from_data


def run_inference(model: PoseFSQAutoEncoder, pose: Pose, only_masked: bool):
    tensor = pose_to_tensor(pose, model.device)
    new_tensor, _ = model(tensor)
    new_pose = pose_from_data(new_tensor[0])

    if only_masked:
        original_pose = pose_from_data(tensor[0])
        mask = pose.body.data.mask
        original_pose.body.data[mask] = new_pose.body.data[mask]
        new_pose = original_pose

    return new_pose


def main():
    parser = argparse.ArgumentParser(description='Reconstruct poses missing keypoints')
    parser.add_argument('--model', type=str, help='Path to trained model', default="sign/mediapipe-vq")
    parser.add_argument('--input', type=str, help='Path to pose file')
    parser.add_argument('--output', type=str, help='Path to output pose file')
    parser.add_argument('--only-masked', action='store_true', help='Only modify masked points')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File {args.input} does not exist")

    with open(args.input, 'rb') as f:
        pose = process_file(f)

    model = load_model(args.model)

    with torch.no_grad():
        new_pose = run_inference(model, pose, args.only_masked)

    with open(args.output, "wb") as f:
        new_pose.write(f)

if __name__ == "__main__":
    main()
