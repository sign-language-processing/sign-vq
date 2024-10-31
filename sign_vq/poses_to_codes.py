import argparse
import json
import zipfile
import csv
from pathlib import Path
from typing import Iterable

import torch
from pose_format import Pose
from tqdm import tqdm

from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std
from sign_vq.model import PoseFSQAutoEncoder


def process_file(file):
    pose = Pose.read(file.read())
    pose = pre_process_mediapipe(pose)
    pose = normalize_mean_std(pose)
    return pose


def load_zip_poses(zip_path: Path) -> Iterable[tuple[str, torch.Tensor]]:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            with zip_ref.open(file) as pose_file:
                yield file, process_file(pose_file)


def load_directory_poses(directory_path: Path) -> Iterable[tuple[str, Pose]]:
    for file in directory_path.glob("*.pose"):
        with open(file, 'rb') as pose_file:
            yield file.name, process_file(pose_file)


def load_poses(data_path: Path) -> Iterable[tuple[str, Pose]]:
    if data_path.is_dir():
        yield from load_directory_poses(data_path)
    elif data_path.suffix == ".zip":
        yield from load_zip_poses(data_path)
    else:
        raise ValueError(f"Unknown data type {data_path}")


def load_model(model_name: str):
    print("Loading model...")
    if Path(model_name).is_dir():
        model_paths = list(Path(model_name).glob("*.ckpt"))
        if len(model_paths) == 0:
            raise ValueError(f"No checkpoint found in {model_name}")
        model_path = sorted(model_paths)[-1]
        args_path = Path(model_name) / "args.json"
    else:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=model_name, filename="model.ckpt")
        args_path = hf_hub_download(repo_id=model_name, filename="args.json")

    with open(args_path, 'r', encoding="utf-8") as f:
        model_args = json.load(f)

    map_location = None if torch.cuda.is_available() else torch.device('cpu')
    model_state = torch.load(model_path, map_location=map_location)["state_dict"]
    model_state = {k.replace("model.", ""): v
                   for k, v in model_state.items()
                   if k.startswith("model.")}
    model = PoseFSQAutoEncoder(**model_args)
    model.load_state_dict(model_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()

    return model

def pose_to_tensor(pose: Pose, device: torch.device):
    tensor = torch.tensor(pose.body.data.filled(0), dtype=torch.float32, device=device)
    # remove person dimension
    tensor = tensor.squeeze(1)
    # add batch dimension
    tensor = tensor.unsqueeze(0)
    return tensor

def run_inference(model: PoseFSQAutoEncoder, poses: Iterable[tuple[str, Pose]], output_path: Path):
    print("Running inference...")
    with open(output_path, 'w', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "fps", "length", "codes"])

        for file, pose in tqdm(poses):
            tensor = pose_to_tensor(pose, model.device)
            codes = model.quantize(tensor)
            codes_list = torch.flatten(codes[0]).tolist()
            writer.writerow([file, pose.body.fps, len(pose.body.data), " ".join(map(str, codes_list))])


def main():
    parser = argparse.ArgumentParser(description='Run inference on a trained model')
    parser.add_argument('--model', type=str, help='Path to trained model',
                        default="sign/mediapipe-vq")
    parser.add_argument('--data', type=str, help='Path to data to run inference on')
    parser.add_argument('--output', type=str, help='Path to output csv file')
    args = parser.parse_args()

    model = load_model(args.model)
    poses = load_poses(Path(args.data))

    with torch.no_grad():
        run_inference(model, poses, Path(args.output))


if __name__ == "__main__":
    main()
