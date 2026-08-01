# UniDepth Robustness Study & ROS2 Integration — Mobile Robotics Final Project

Raja Haseeb Ahmad | 2025-MS-MC-01 | MS Mechatronics Engineering

## Overview
Reproduction and extension of UniDepth V2 for outdoor mobile robot depth
perception, including a 5-condition environmental degradation study
(rain, fog, low light, motion blur, sensor noise) and a model-side
self-distillation fine-tuning solution, evaluated via a held-out
protocol. A ROS2 Humble + Docker deployment pipeline is included to
serve model checkpoints to a live camera topic.

Full details, methodology, and results are in the accompanying IEEE
paper (`paper/`).

## Repository Structure
```
├── paper/
│   ├── Haseeb_MR_Unidepth_IEEE_Paper.tex   # LaTeX source
│   └── Haseeb_MR_Unidepth_IEEE_Paper.pdf   # Compiled paper
├── notebook/
│   └── MR_Project_Final.ipynb              # Full Colab pipeline: baseline,
│                                             # degradation, fine-tuning, eval
├── ros2_ws/
│   ├── image_publisher.py                  # Publishes camera frames to
│                                             # /camera/image_raw
│   └── depth_node.py                       # Subscribes to camera topic,
│                                             # publishes depth to /unidepth/depth
├── results/
│   └── (figures, CSVs, checkpoints referenced in the paper)
└── README.md
```

## Dependencies

**Colab / Model pipeline:**
- Python 3.12
- PyTorch (CUDA build)
- UniDepth V2 (`pip install git+https://github.com/lpiccinelli-eth/UniDepth.git`)
- timm, einops, opencv-python, scikit-image, torchvision, numpy, pandas

**ROS2 pipeline:**
- ROS2 Humble Hawksbill (via Docker image `ros:humble`)
- Docker Desktop (Windows/Mac) or native Docker (Linux)
- rclpy, sensor_msgs (included in ros:humble image)
- Python packages inside container: `pillow`, `numpy`

## Installation & Setup

### 1. Model pipeline (Google Colab)
Open `notebook/MR_Project_Final.ipynb` in Google Colab, run cells
top-to-bottom. Requires a GPU runtime (T4 or better).

### 2. ROS2 pipeline (local machine, Docker)
```bash
# Pull the ROS2 Humble image
docker pull ros:humble

# Run a container with the ros2_ws folder mounted
docker run -it --name unidepth_robot -v <path-to-ros2_ws>:/workspace ros:humble bash

# Inside the container:
apt update && apt install -y python3-pip
pip3 install pillow numpy
source /opt/ros/humble/setup.bash

# Terminal 1 — publisher
python3 /workspace/image_publisher.py --export-dir /workspace/<data-folder>

# Terminal 2 (docker exec -it unidepth_robot bash, then source setup.bash again)
python3 /workspace/depth_node.py --export-dir /workspace/<data-folder>
```

## Sample Results
See `results/` and the paper's Table I (held-out validated improvement)
and Table II (single-scene demonstration) for full quantitative results.
Summary: model-side fine-tuning shows consistent improvement for sensor
noise (+6.5% held-out), with flat-to-mildly-negative effects for rain,
fog, low light, and motion blur at this data scale — see the paper's
Discussion (Section VI) and Challenges (Section IX) for full analysis.

## Known Limitations
- Fine-tuning uses 21 training / 9 held-out KITTI frames from a single
  drive sequence (not the full dataset).
- A coordinate-space mismatch between training-time and inference-time
  resolution handling was identified late in development and is flagged
  as an open validity concern in the paper (Section IX).
- ROS2 pipeline has not been validated on physical or simulated robot
  hardware-in-the-loop; it has been verified for correct topic
  publish/subscribe message passing only.

## License / Academic Use
Submitted as coursework for the Mobile Robotics Systems Exploration
Project, MS Mechatronics Engineering.
