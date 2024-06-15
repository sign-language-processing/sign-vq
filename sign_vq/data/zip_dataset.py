import io
import zipfile
from pathlib import Path

import numpy as np
from pose_format import Pose
from tqdm import tqdm

from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std


def save_poses_to_zip(directory: str, zip_filename: str):
    pose_files = list(Path(directory).glob("*.pose"))
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in tqdm(pose_files):
            with open(file, 'rb') as pose_file:
                pose = Pose.read(pose_file.read())
                pose = pre_process_mediapipe(pose)
                pose = normalize_mean_std(pose)

                # Using the file name as the zip entry name
                npz_filename = file.stem + '.npz'

                # Saving the masked array to a temporary buffer
                with io.BytesIO() as buf:
                    data = pose.body.data[:, 0, :, :]  # only first person

                    float16_data = data.filled(0).astype(np.float16)
                    np.savez_compressed(buf, data=float16_data, mask=data.mask)
                    zip_file.writestr(npz_filename, buf.getvalue())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--dir', type=str, help='Directory containing the pose files')
    parser.add_argument('--out', type=str, help='Output zip file')

    args = parser.parse_args()

    save_poses_to_zip(args.dir, args.out)
