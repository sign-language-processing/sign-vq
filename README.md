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

# 2. Creates a ZIP file of the poses after normalizing them. (about 45GB)
sbatch scripts/zip_dataset.sh "$DATA_DIR/sign-mt-poses" "$DATA_DIR/normalized.zip"

# 3. Trains the model and reports to `wandb`.
sbatch scripts/train_model.sh "$DATA_DIR/normalized.zip"
```

## Training Output

In Weights & Biases, we can see the training progress.
In validation, we generate a video from the compressed poses (right) and compare it to the original video (left).

(Current examples are from a model currently training, in its 3rd epoch)

| 0                                       | 1                                       | 2                                       | 3                                       | 4                                       |
|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| ![](assets/validation/validation_0.gif) | ![](assets/validation/validation_1.gif) | ![](assets/validation/validation_2.gif) | ![](assets/validation/validation_3.gif) | ![](assets/validation/validation_4.gif) |

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