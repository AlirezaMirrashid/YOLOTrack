# YOLOTrack (YOLO + DeepSORT Object Detection and Tracking)
This repository demonstrates a video processing pipeline that uses **YOLO** (via Ultralytics) for object detection and **DeepSORT** for object tracking. Users can now select which object labels to track by supplying a comma-separated list via the command line.

## Overview

- **YOLODetector**: Loads a YOLO model and detects objects in video frames. It filters the detections based on the labels provided by the user.
- **DeepSortTracker**: Uses the DeepSORT algorithm (via the `deep_sort_realtime` package) to assign unique IDs and track objects across frames.
- **VideoProcessor**: Captures video from a file or webcam, processes each frame with detection and tracking, and displays the results in real time.

## Features

- Real-time object detection and tracking.
- User-selectable object labels (e.g., `car`, `person`, etc.) via a command-line argument.
- Visual output with bounding boxes, labels, detection confidence, and track IDs.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/YOLO-DeepSORT-Tracker.git
   cd YOLO-DeepSORT-Tracker
Create a virtual environment (optional but recommended):


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required packages:

Ensure you have pip installed, then run:

bash
pip install -r requirements.txt
A sample requirements.txt might include:

txt
opencv-python
torch
ultralytics
deep_sort_realtime
(Adjust package versions as needed.)

Usage
Run the main script using Python. You can pass command-line arguments to select the video source, YOLO model weights, and the labels you wish to track.

Example Commands
Using your webcam (default index 0) and tracking cars:

bash
python your_script.py --video 0 --model yolo11n.pt --labels car
Using a video file and tracking multiple objects (e.g., cars and persons):

bash

python your_script.py --video path/to/video.mp4 --model yolo11n.pt --labels car,person
Command-Line Arguments
--video: Path to a video file or camera index (default: 0 for webcam).
--model: Path to the YOLO model weights file (default: yolo11n.pt).
--labels: Comma-separated list of object labels to track (default: car).

How It Works
Detection:
The YOLODetector class loads the YOLO model and processes each frame. It filters the detections to include only the labels specified by the user.

Tracking:
The DeepSortTracker class converts the detections into the format required by DeepSORT and updates the tracker to maintain object identities across frames.

Display:
The VideoProcessor class reads video frames, applies detection and tracking, and overlays bounding boxes, labels, detection confidences, and unique track IDs on the frame. Press q to exit.

License
This project is licensed under the MIT License.
