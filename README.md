# Object Detection Realtime Project
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/object_dectection_realtime_project?style=flat-square)
![Python](https://img.shields.io/badge/Python-75.3%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/object_dectection_realtime_project?style=flat-square)

This repository contains the complete codebase for Jidapa's Deep Learning-based Object Detection Realtime Project.

## 📌 Overview

This project implements **real-time object detection** on video feeds using the lightweight and fast **YOLOv8n** model. It enables continuous object recognition via live CCTV streams or screen capture, offering practical applications in **security**, **monitoring**, and **automated surveillance**. With efficient processing and minimal latency, it is designed for deployment on accessible hardware, including **NVIDIA Tesla T4** or similar.

### 🧩 Problem Statement

Monitoring environments via CCTV or screen feeds often requires human oversight, which can be inefficient and error-prone. This project automates the detection of key objects in real-time, enhancing the effectiveness of surveillance systems while reducing operational cost and human workload.

### 🔍 Approach

- A custom Python application was developed to capture frames from CCTV video feeds.
- These frames are passed to a fine-tuned **YOLOv8n** model using the `ultralytics` library.
- Detected objects are displayed in real time using `cv2.imshow`, with adjustable thresholds and output overlays.

### 🎢 Processes

1. **Frame Capture** – Use OpenCV to continuously grab frames from a camera or screen.
2. **Model Loading** – Load YOLOv8n with pretrained weights using the Ultralytics interface.
3. **Realtime Detection** – Run inference on each frame and visualize bounding boxes with class labels.
4. **Extended Features** – Implement utilities such as post-processing, FPS tracking, and flexible input handling in `utils/realtime_extended.py`.
5. **Demo & Evaluation** – Provide example results via a notebook and video demonstration.

### 🎯 Results & Impact

- Achieved **high frame rate object detection** suitable for live video streams.
- Delivered consistent detection accuracy while maintaining **real-time latency** (~5–15ms/frame on T4).
- Demonstrated potential for deployment in **retail monitoring**, **home security**, or **industrial surveillance** scenarios.

### ⚙️ Model Development Challenges

- **Frame Rate Optimization** – Managed CPU/GPU balance to maintain low latency.
- **Detection Thresholds** – Tuned confidence thresholds to reduce false positives in dynamic environments.
- **Hardware Compatibility** – Ensured CUDA support fallback for diverse hardware configurations.
- **Stream Stability** – Addressed frame drop and buffer handling issues in unstable network CCTV streams.


## Libraries and Tools Used
- **Data Analysis**: `pandas`, `NumPy` for data manipulation and preprocessing.
- **Image Processing**: `opencv-python` (`cv2`) to handle video frame extraction and display.
- **Pretrained Model**: YOLOv8n, the lightweight YOLOv8 architecture optimized for realtime performance.
- **Deep Learning Frameworks**: `pytorch`, `ultralytics`, and `yolo` for model loading, inference, and utilities.

## Repository Structure
- `main.py` The main application script that initializes the video capture from CCTV, loads the YOLOv8n model, processes video frames, and displays realtime detection results.
- `utils/realtime_extended.py` Contains utility functions and extended features such as advanced frame processing, detection postprocessing, and video stream handling.
- `requirements.txt` Lists all Python dependencies and versions required to run the project smoothly.
- `example_result_object_detection_realtime.ipynb` Jupyter notebook demonstrating example outputs from a demo run.
- [`example_result video`](https://drive.google.com/file/d/1unFScKpaFGszicRZX8QKoDSZvKGct_dj/preview) Video demonstrating example outputs from a demo run  

## Usage Notes
- Ensure that your video input device (CCTV or video file) is correctly configured in `main.py`.
- The model weights are loaded automatically via the `ultralytics` YOLOv8n interface; no manual download required.
- Real-time performance depends on your hardware — a CUDA-enabled GPU significantly accelerates detection.
- Customize detection confidence thresholds and other parameters in `main.py` as needed.

## Getting Started
1. Clone the repo
   
   ```bash
   git clone https://github.com/JPP-J/object_dectection_realtime_project.git
   ```
   
   ```bash 
   cd object_dectection_realtime_project
   ```
3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
5. Run realtime detection  
   ```bash
   python main.py
   ```

*For more detailed setup and troubleshooting, please refer to the notebook and code comments.*

---

