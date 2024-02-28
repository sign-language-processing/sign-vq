# Sign MediaPipe VQ

We try to compress mediapipe poses using VQ-VAE and then generate videos from the compressed poses.
Given a good quantizer, we can use it for downstream tasks like SignWriting transcription or animation.

## Training a model

```bash
# 0. Setup the environment.
conda create --name vq python=3.11
conda activate vq
pip install .

# 1. Downloads lots of poses from the bucket. (about 508GB)
DATA_DIR=/scratch/amoryo/poses
sbatch scripts/sync_bucket.sh "$DATA_DIR/sign-mt-poses"

POSES_DIR=/shares/volk.cl.uzh/amoryo/datasets/sign-mt-poses

# 2. Creates a ZIP file of the poses after normalizing them. (about 45GB)
sbatch scripts/zip_dataset.sh "$POSES_DIR" "$DATA_DIR/normalized.zip"

# 3. Trains the model and reports to `wandb`.
sbatch scripts/train_model.sh "$DATA_DIR/normalized.zip"
```

## Training Output

In Weights & Biases, we can see the training progress.
In validation, we generate a video from the compressed poses (right) and compare it to the original video (left).
(This is the output using 4 codebooks of size 1024.)

| 0                                       | 1                                       | 2                                       | 3                                       | 4                                       |
|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| ![](assets/validation/validation_0.gif) | ![](assets/validation/validation_1.gif) | ![](assets/validation/validation_2.gif) | ![](assets/validation/validation_3.gif) | ![](assets/validation/validation_4.gif) |

## Inference

To quantize a pose file, directory of poses, or Zip file of poses, use the `inference` command.
```bash
poses_to_codes --data="DIRECTORY" --output="output.csv"
```

To convert codes back to poses, use the `codes_to_poses` command.
```bash
codes_to_poses --output="DIRECTORY" --data="codes_file.txt" 
# Or directly from codes, 5 frames example
codes_to_poses --output="test.pose" --codes="731 63 540 261 787 63 250 100 492 351 530 307 939 63 532 61 788 55 530 60"
```

## Background

Vector Quantization has been successfully used by many for highly compressing images and audio.
For example, by Deepmind and OpenAI for high quality generation of images (VQ-VAE-2) and music (Jukebox).

We use a Finite Scalar Quantization.
This work out of Google Deepmind aims to vastly simplify the way vector quantization is done for generative modeling,
removing the need for commitment losses, EMA updating of the codebook, as well as tackle the issues with codebook
collapse or insufficient utilization. They simply round each scalar into discrete levels with straight through
gradients; the codes become uniform points in a hypercube.

## Data

Data is expected as a zip file of numpy masked arrays.
See [sign_vq/data/README.md](sign_vq/data/README.md) for more details.

## Other Resources

- [MotionGPT](https://github.com/OpenMotionLab/MotionGPT): Human Motion as a Foreign Language
- [T2M-GPT](https://github.com/Mael-zys/T2M-GPT): Generating Human Motion from Textual Descriptions with Discrete
  Representations

## Recent Updates

- 2024-02-25: 
  - Update `vector_quantize_pytorch` to 1.14.1
  - Increase steps from 1e6 to 3e6
- 2024-02-26:
  - Hide body wrists to avoid flickering (only use hand wrists)
- 2024-02-28:
  - Bring wrists back

- Next?
  - Increase learning rate to 5e-3
  - Increase transformer layers to 8
  - Increase hidden dimension to 1024