from pathlib import Path

import cv2
import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import reduce_holistic

from sign_vq.data.normalize import load_mean_and_std, unshift_hand, load_pose_header

if __name__ == "__main__":
    mean, _ = load_mean_and_std()

    data = mean.reshape((1, 1, -1, 3)) * 1000
    confidence = np.ones((1, 1, len(mean)))
    body = NumPyPoseBody(fps=1, data=data, confidence=confidence)
    pose = Pose(header=load_pose_header(), body=body)

    unshift_hand(pose, "RIGHT_HAND_LANDMARKS")
    unshift_hand(pose, "LEFT_HAND_LANDMARKS")

    poses = {
        "full": pose,
        "reduced": reduce_holistic(pose)
    }
    for name, pose in poses.items():
        pose.focus()

        v = PoseVisualizer(pose)
        image_path = Path(__file__).parent / f"mean_pose_{name}.png"
        cv2.imwrite(str(image_path), next(v.draw()))
