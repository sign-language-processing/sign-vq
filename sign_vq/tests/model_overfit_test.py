import random
import unittest
from unittest.mock import MagicMock

import torch
from pose_format.torch.masked import MaskedTensor
from pose_format.torch.masked.collator import zero_pad_collator
from tqdm import tqdm

from sign_vq.model import PoseFSQAutoEncoder, AutoEncoderLightningWrapper


def get_batch(bsz=4):
    data_tensor = torch.tensor([[[1, 1]], [[2, 2]], [[3, 3]]], dtype=torch.float32)
    return {
        "text": ["text1"] * bsz,
        "pose": {
            "length": torch.tensor([3], dtype=torch.float32).expand(bsz, 1),
            "data": data_tensor.expand(bsz, *data_tensor.shape),
            "confidence": torch.ones([bsz, 3, 1]),
            "inverse_mask": torch.ones([bsz, 3]),
        },
    }


pose_dim = (2, 3)


class ModelOverfitTestCase(unittest.TestCase):
    def model_setup(self):
        model = PoseFSQAutoEncoder(codebook_size=2 ** 4, num_codebooks=1, pose_dims=pose_dim,
                                   hidden_dim=16, nhead=2, num_layers=2, dim_feedforward=32)
        model = AutoEncoderLightningWrapper(model, learning_rate=5e-2, warmup_steps=1)
        model.log = MagicMock(return_value=True)
        return model

    def test_model_should_overfit(self):
        torch.manual_seed(42)
        random.seed(42)

        poses = [
            MaskedTensor(torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)),
            MaskedTensor(torch.tensor([[[3, 2, 1], [0, -1, -2]]], dtype=torch.float32)),
        ]
        batch = zero_pad_collator(poses)
        print("batch", batch.shape)

        model = self.model_setup()

        optimizers = model.configure_optimizers()
        optimizer = optimizers["optimizer"]
        scheduler = optimizers["lr_scheduler"]["scheduler"]

        model.train()
        torch.set_grad_enabled(True)

        # Simple training loop
        losses = []
        for _ in tqdm(range(70)):
            optimizer.zero_grad()  # clear gradients

            loss = model.training_step(batch)
            loss_float = float(loss.detach())
            losses.append(loss_float)

            loss.backward()  # backward
            optimizer.step()  # update parameters
            scheduler.step()  # update learning rate

        print("losses", losses)
        print("last loss", losses[-1])

        model.eval()
        with torch.no_grad():
            prediction, _ = model(batch)

        print("batch", batch.tensor)
        print("prediction", prediction)
        print("torch.round(prediction)", torch.round(prediction))

        self.assertEqual(batch.shape, prediction.shape)
        self.assertTrue(torch.all(torch.eq(torch.round(prediction), batch.tensor)))


if __name__ == '__main__':
    unittest.main()
