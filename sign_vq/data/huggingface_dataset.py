import argparse
from pathlib import Path

import numpy as np

import datasets
from pose_format import Pose

from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std


class SignLanguagePoseDataset(datasets.GeneratorBasedBuilder):
    def __init__(self, pose_directory: Path):
        super().__init__()

        self.data = list(pose_directory.glob("*.pose"))

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "data": datasets.Array3D(dtype="float16", shape=(None, 178, 3)),
                    "mask": datasets.Array3D(dtype="bool", shape=(None, 178, 3)),
                }
            )
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={})
        ]

    def _generate_examples(self, **unused_kwargs):
        for i, file in enumerate(self.data):
            with open(file, 'rb') as pose_file:
                pose = Pose.read(pose_file.read())
                pose = pre_process_mediapipe(pose)
                pose = normalize_mean_std(pose)

            data = pose.body.data[:, 0, :, :]  # only first person

            float16_data = data.filled(0).astype(np.float16)
            if i == 0:
                print(float16_data.shape, float16_data.dtype, data.mask.shape, data.mask.dtype)

            yield i, {
                "data": float16_data,
                "mask": data.mask,
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    pose_directory = Path(args.directory)
    output_path = Path(args.output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    dataset = SignLanguagePoseDataset(pose_directory)
    dataset.download_and_prepare(output_path)
    dataset.as_dataset().save_to_disk(output_path)


if __name__ == "__main__":
    main()
