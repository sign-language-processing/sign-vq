import unittest
from unittest.mock import MagicMock

import torch

from pose_format.torch.masked import MaskedTensor

from sign_vq.model import PoseFSQAutoEncoder, AutoEncoderLightningWrapper


class ModelTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_dim = (2, 2)
        self.seq_length = 5

    def model_setup(self):
        model = PoseFSQAutoEncoder(codebook_size=2 ** 4, num_codebooks=2, pose_dims=self.pose_dim,
                                   hidden_dim=16, nhead=2, num_layers=2, dim_feedforward=32)
        model = AutoEncoderLightningWrapper(model)
        model.log = MagicMock(return_value=True)
        return model

    def test_forward_yields_same_shape(self):
        model = self.model_setup()
        pose = MaskedTensor(torch.full((4, 3, *self.pose_dim), fill_value=2, dtype=torch.float))
        out_pose, _ = model(pose)

        self.assertEqual(pose.shape, out_pose.shape)

    def test_training_step_expected_loss_zero(self):
        model = self.model_setup()
        tensor = torch.full((4, 3, *self.pose_dim), fill_value=2, dtype=torch.float)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        batch = MaskedTensor(tensor, mask)

        loss = float(model.training_step(batch))
        self.assertEqual(0, loss)

    def test_training_step_expected_loss_finite(self):
        model = self.model_setup()
        batch = MaskedTensor(torch.full((4, 3, *self.pose_dim), fill_value=2, dtype=torch.float))

        loss = model.training_step(batch)
        self.assertNotEqual(0, float(loss))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
