# Object Detection Realtime Project
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/object_dectection_realtime_project?style=flat-square)
![Python](https://img.shields.io/badge/Python-75.3%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/object_dectection_realtime_project?style=flat-square)

This repository contains the complete codebase for Jidapa's Deep Learning-based Object Detection Realtime Project.

## Project Overview
- **Description**: This project captures screen recordings or live video streams from personal CCTV cameras via a custom application and performs realtime object detection using the YOLOv8n model. It enables monitoring and automatic recognition of objects in video feeds, making it useful for security and surveillance applications.
- **Goal**: Achieve efficient and accurate realtime object detection with minimal latency on accessible hardware like NVIDIA Tesla T4 GPUs or equivalent.

## Libraries and Tools Used
- **Data Analysis**: `pandas`, `NumPy` for data manipulation and preprocessing.
- **Image Processing**: `opencv-python` (`cv2`) to handle video frame extraction and display.
- **Pretrained Model**: YOLOv8n, the lightweight YOLOv8 architecture optimized for realtime performance.
- **Deep Learning Frameworks**: `pytorch`, `ultralytics`, and `yolo` for model loading, inference, and utilities.

## Repository Structure
- `main.py` The main application script that initializes the video capture from CCTV, loads the YOLOv8n model, processes video frames, and displays realtime detection results.
- `utils/realtime_extended.py` Contains utility functions and extended features such as advanced frame processing, detection postprocessing, and video stream handling.
- `requirements.txt` Lists all Python dependencies and versions required to run the project smoothly.
- `example_result_object_detection_realtime.ipynb` Jupyter notebook demonstrating example outputs, performance metrics, and visualization from a demo run.

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

*Feel free to contribute, open issues, or ask questions via GitHub.*

