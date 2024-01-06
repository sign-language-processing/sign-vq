import cv2
import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer

from sign_vq.data.normalize import load_mean_and_std, unshift_hand, load_pose_header

if __name__ == "__main__":
    mean, _ = load_mean_and_std()

    data = mean.reshape(shape=(1, 1, -1, 3)) * 1000
    confidence = np.ones((1, 1, len(mean)))
    body = NumPyPoseBody(fps=1, data=data, confidence=confidence)
    pose = Pose(header=load_pose_header(), body=body)

    unshift_hand(pose, "RIGHT_HAND_LANDMARKS")
    unshift_hand(pose, "LEFT_HAND_LANDMARKS")

    pose.focus()

    v = PoseVisualizer(pose)
    cv2.imwrite("mean_pose.png", next(v.draw()))
