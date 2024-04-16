from typing import Union

import cv2
import numpy as np
import torch
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.torch.masked import MaskedTensor
from torch import Tensor

from sign_vq.data.normalize import load_pose_header, unnormalize_mean_std, unshift_hand


def pose_from_data(pose_data: Union[MaskedTensor, Tensor]):
    from pose_format.numpy import NumPyPoseBody

    if isinstance(pose_data, Tensor):
        pose_data = MaskedTensor(pose_data)

    # Add person dimension
    pose_data.tensor = pose_data.tensor.unsqueeze(1)
    pose_data.mask = pose_data.mask.unsqueeze(1)

    if pose_data.dtype != torch.float32:
        pose_data.tensor = pose_data.tensor.to(torch.float32)

    np_data = pose_data.tensor.numpy()
    np_confidence = pose_data.mask.numpy().astype(np.float32).max(-1)
    np_body = NumPyPoseBody(fps=25, data=np_data, confidence=np_confidence)

    pose = Pose(header=load_pose_header(), body=np_body)
    pose = unnormalize_mean_std(pose)
    unshift_hand(pose, "RIGHT_HAND_LANDMARKS")
    unshift_hand(pose, "LEFT_HAND_LANDMARKS")

    # Resize pose
    new_width = 200
    shift = 1.25
    shift_vec = np.full(shape=(pose.body.data.shape[-1]), fill_value=shift, dtype=np.float32)
    pose.body.data = (pose.body.data + shift_vec) * new_width
    pose.header.dimensions.height = pose.header.dimensions.width = int(new_width * shift * 2)

    return pose


def draw_pose(pose_data: MaskedTensor):
    pose = pose_from_data(pose_data)

    # Draw pose
    visualizer = PoseVisualizer(pose)
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in visualizer.draw()]
    return np.stack(frames)


def draw_original_and_predicted_pose(original: MaskedTensor, predicted: Tensor):
    original = MaskedTensor(original.tensor.cpu(), original.mask.cpu())
    predicted = predicted.cpu()

    # to find the pose length, find the last frame where the confidence is not zero
    frame_confidence = original.mask.numpy().max(-1).max(-1)  # (frames)
    pose_length = frame_confidence.nonzero()[0].max() + 1

    original = original[:pose_length]
    predicted = MaskedTensor(predicted[:pose_length])

    original_video = draw_pose(original)
    predicted_video = draw_pose(predicted)
    return np.concatenate([original_video, predicted_video], axis=2)


if __name__ == "__main__":
    fake_pose = MaskedTensor(torch.zeros(size=(100, 178, 3), dtype=torch.float32))
    draw_pose(fake_pose)
    # video = draw_original_and_predicted_pose(fake_pose, fake_pose.tensor)
    # print(video.shape)
    # print(np.moveaxis(video, 3, 1).shape)
