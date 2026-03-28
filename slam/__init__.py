"""
slam — RGB-D Visual SLAM library.

Core modules:
    camera          Pinhole camera model and projection utilities.
    dataset         TUM RGB-D dataset loader.
    frame           Frame dataclass (RGB, depth, pose, features).
    se3             SE(3) / SO(3) Lie-group utilities.
    vo_frontend     ORB-based visual odometry front-end.
    loop_closure    Descriptor-based loop-closure detection and PnP verification.
    pose_graph      Pose-graph data structures and Gauss-Newton optimizer.
    map_management  Keyframe map and pose-graph manager.
    slam_hparams    Hyperparameter dataclass (tuned by the RL agent).
    slam_runner     Top-level SLAM pipeline callable.
    evaluation      ATE / RMSE trajectory evaluation helpers.
    visualization   Matplotlib trajectory and pose-graph plots.
"""

from .camera import PinholeCamera, freiburg2_camera
from .dataset import TUMRGBDDataset, RGBDFrame
from .frame import Frame
from .se3 import hat3, vee3, so3_exp, so3_log, se3_exp, se3_log
from .slam_hparams import SlamHyperParams

__all__ = [
    "PinholeCamera",
    "freiburg2_camera",
    "TUMRGBDDataset",
    "RGBDFrame",
    "Frame",
    "hat3",
    "vee3",
    "so3_exp",
    "so3_log",
    "se3_exp",
    "se3_log",
    "SlamHyperParams",
]
