[README.md](https://github.com/user-attachments/files/26321371/README.md)
# RGB-D SLAM with Deep RL Hyperparameter Optimization

A from-scratch Python implementation of **keyframe-based RGB-D Visual SLAM** enhanced with a **Deep Reinforcement Learning (DRL) agent** that automatically tunes the system's hyperparameters per sequence — replacing manual, trial-and-error calibration.

Evaluated on the [TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) (`freiburg2_pioneer_slam2`).

---

## Overview

Traditional SLAM systems expose many brittle hyperparameters (keyframe thresholds, loop-closure settings, RANSAC tolerances) that are typically hand-tuned for each dataset. This project learns those parameters automatically using a **REINFORCE policy-gradient agent**.

### Key ideas

| Component | Approach |
|---|---|
| Visual Odometry | ORB features + Lowe's ratio test + PnP RANSAC |
| Loop Closure | Descriptor match-count candidate search + geometric PnP verification |
| Pose-Graph Optimization | Gauss-Newton on SE(3) with numeric Jacobians |
| RL Environment | 1-step episodic MDP; reward = −ATE RMSE − density penalties |
| RL Algorithm | REINFORCE with a diagonal Gaussian policy (PyTorch) |

---

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                   SLAM Pipeline                       │
│                                                       │
│  RGB-D frames ──► VO Front-end (ORB + PnP RANSAC)    │
│                        │                              │
│                   Keyframe selection                  │
│                  (trans / rot thresholds)             │
│                        │                              │
│              Loop-closure detection                   │
│           (descriptor search + PnP verify)            │
│                        │                              │
│            Pose-graph optimization                    │
│              (Gauss-Newton, SE(3))                    │
│                        │                              │
│                  ATE Evaluation  ──► reward           │
└───────────────────────────────────────────────────────┘
                         ▲
                         │  hyperparameters (6-D action)
                         │
┌───────────────────────────────────────────────────────┐
│              RL Agent (REINFORCE)                     │
│                                                       │
│   obs (4-D segment descriptor)                        │
│        ──► PolicyNet (2-layer MLP, Gaussian)          │
│        ──► action ∈ [-1,1]^6                          │
│        ──► mapped to concrete hyperparams             │
└───────────────────────────────────────────────────────┘
```

### Tuned Hyperparameters

| Action dim | Parameter | Range |
|---|---|---|
| a[0] | `trans_thresh` | 0.05 – 0.50 m |
| a[1] | `rot_thresh_deg` | 2 – 20 ° |
| a[2] | `min_frame_gap` | 5 – 30 frames |
| a[3] | `lc_min_frame_separation` | 30 – 200 frames |
| a[4] | `lc_min_inliers` | 20 – 200 |
| a[5] | `lc_pnp_reproj_thresh` | 0.5 – 5.0 px |

---

## Results

Evaluated on `rgbd_dataset_freiburg2_pioneer_slam2` (54 segments, ~720 frames each).

| Configuration | ATE RMSE | Keyframes | Loop closures |
|---|---|---|---|
| Baseline (defaults) | ~0.45 m | ~35 | ~8 |
| **DRL-optimized** | **~0.34 m** | **21** | **16** |

The DRL agent learned to use **sparser keyframes** and **more aggressive loop closure**, achieving a **~25% reduction in trajectory error**.

---

## Repository Structure

```
.
├── slam/                    # Core SLAM library (pure Python + NumPy + OpenCV)
│   ├── camera.py            # Pinhole camera model
│   ├── dataset.py           # TUM RGB-D dataset loader
│   ├── evaluation.py        # ATE / RMSE evaluation (Umeyama alignment)
│   ├── frame.py             # Frame dataclass
│   ├── loop_closure.py      # Loop-closure detection and verification
│   ├── map_management.py    # Keyframe map
│   ├── pose_graph.py        # Pose-graph data structures + GN optimizer
│   ├── se3.py               # SE(3) / SO(3) Lie-group math
│   ├── slam_hparams.py      # Hyperparameter dataclass
│   ├── slam_runner.py       # Top-level SLAM pipeline
│   └── visualization.py    # Trajectory and pose-graph plots
│
├── rl/                      # Reinforcement learning components
│   ├── env.py               # Gym-style 1-step SLAM environment
│   ├── policy.py            # Gaussian policy network (PyTorch)
│   └── train.py             # REINFORCE training loop
│
├── scripts/
│   ├── run_baseline_slam.py # Run SLAM with fixed / JSON hyperparameters
│   └── run_drl_slam.py      # Train RL agent + evaluate best config
│
├── configs/
│   ├── default_hparams.json # Default hyperparameters
│   └── best_drl_hparams.json# Best config found by DRL
│
├── assets/                  # Images used in this README
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/<your-username>/slam-drl-hyperopt.git
cd slam-drl-hyperopt
pip install -r requirements.txt
```

Python 3.10+ is required.

---

## Dataset Setup

Download a TUM RGB-D sequence and optionally generate an `associate.txt` file:

```bash
# Example: freiburg2_pioneer_slam2
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam2.tgz
tar xzf rgbd_dataset_freiburg2_pioneer_slam2.tgz

# (Optional) generate associate.txt for faster loading
python -c "
import associate
associate.main(['rgb.txt', 'depth.txt', '0.02', 'associate.txt'])
" -d rgbd_dataset_freiburg2_pioneer_slam2
```

The dataset loader auto-discovers `associate.txt` if present; otherwise it
performs on-the-fly nearest-timestamp association.

---

## Usage

### Baseline SLAM (fixed hyperparameters)

```bash
python scripts/run_baseline_slam.py \
    --dataset /path/to/rgbd_dataset_freiburg2_pioneer_slam2 \
    --save-plots

# Override specific hyperparameters via JSON
python scripts/run_baseline_slam.py \
    --dataset /path/to/... \
    --hparams configs/best_drl_hparams.json \
    --save-plots
```

### DRL Hyperparameter Search

```bash
python scripts/run_drl_slam.py \
    --dataset /path/to/rgbd_dataset_freiburg2_pioneer_slam2 \
    --episodes 100 \
    --save-dir outputs/drl
```

The script trains the RL agent, saves the best policy to `outputs/drl/slam_rl_policy_best.pt`,
and runs a final evaluation with the best-found hyperparameters.

---

## How It Works

### SLAM Pipeline

1. **Visual Odometry** — detects ORB keypoints, back-projects them to 3-D using the
   depth map, matches consecutive frames, and estimates the relative pose via
   PnP RANSAC (`cv2.solvePnPRansac`).

2. **Keyframe Selection** — a new keyframe is inserted when the camera has moved
   more than `trans_thresh` meters *or* rotated more than `rot_thresh_deg` degrees
   since the last keyframe, subject to a minimum frame gap.

3. **Loop Closure** — for each new keyframe, past keyframes are ranked by ORB
   descriptor match count. Top candidates undergo geometric verification (PnP RANSAC).
   Verified loops are added to the pose graph as loop edges.

4. **Pose-Graph Optimization** — a Gauss-Newton optimizer minimizes the SE(3)
   residuals across all odometry and loop edges. Jacobians are computed numerically.

5. **Evaluation** — trajectory positions are aligned to TUM ground truth via
   Umeyama alignment, and ATE RMSE is computed.

### RL Formulation

- **State** — a 4-D normalized descriptor of the current segment
  (start/end indices, length, number of segments).
- **Action** — a 6-D continuous vector in [-1, 1]⁶ mapped to SLAM hyperparameters.
- **Reward** — `−RMSE − λ_kf × KF_density − λ_lc × LC_density`
  (penalizes both trajectory error and computational overhead).
- **Algorithm** — REINFORCE with a diagonal Gaussian policy trained with Adam.

Each episode runs the entire SLAM pipeline on one randomly sampled segment and
receives a scalar reward. Training requires ~50–100 episodes to converge on
a single sequence.

---

## Extending the Project

- **New datasets** — subclass `TUMRGBDDataset` or implement the same
  `__len__` / `__getitem__` interface returning `RGBDFrame` objects.
- **New features** — swap ORB for SuperPoint / SIFT in `VisualOdometry._extract_features`.
- **New RL algorithms** — the `SlamHyperParamEnv` follows the Gym API;
  any off-policy or on-policy algorithm can be plugged in via `rl/train.py`.
- **Multi-sequence training** — pass multiple dataset roots to `SlamHyperParamEnv`
  to generalize the policy across sequences.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{shah2025slamdrl,
  author = {Nitai Shah},
  title  = {RGB-D SLAM with Deep Reinforcement Learning Hyperparameter Optimization},
  year   = {2025},
  url    = {https://github.com/<your-username>/slam-drl-hyperopt}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
