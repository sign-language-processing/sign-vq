import argparse
import os
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from sign_vq.model import PoseFSQAutoEncoder
from sign_vq.poses_to_codes import load_model
from sign_vq.utils import pose_from_data


def run_inference(model: PoseFSQAutoEncoder, codes: Iterable[str], output_path: Path):
    for i, line in enumerate(tqdm(codes)):
        int_codes = torch.tensor([[int(c) for c in line.split(" ")]], dtype=torch.long, device=model.device)
        poses_data = model.unquantize(int_codes)
        pose = pose_from_data(poses_data[0])

        pose_output_path = output_path / f"{i}.pose" if output_path.is_dir() else output_path

        with open(pose_output_path, "wb") as f:
            pose.write(f)


def main():
    parser = argparse.ArgumentParser(description='Run inference on a trained model')
    parser.add_argument('--model', type=str, help='Path to trained model', default="sign/mediapipe-vq")
    parser.add_argument('--codes', type=str,
                        help='Codes or path to text file with codes, new line separated')
    parser.add_argument('--output', type=str, help='Path to output pose file / directory')
    args = parser.parse_args()

    if os.path.exists(args.codes):
        with open(args.codes, 'r', encoding='utf-8') as f:
            codes = f.read().splitlines()
        assert os.path.isdir(args.output), "When codes is a file, output must be a directory"
    else:
        codes = [args.codes]
        assert not os.path.isdir(args.output), "When codes is given directly, output must be a file"

    assert len(codes) > 0, "No codes found"

    model = load_model(args.model)

    with torch.no_grad():
        run_inference(model, codes, Path(args.output))


if __name__ == "__main__":
    main()
