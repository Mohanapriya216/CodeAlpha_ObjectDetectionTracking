# Object Detection and Tracking

## Overview

This project implements an object detection and tracking application using Python, OpenCV, YOLO, Deep SORT, and Streamlit. It allows users to upload a video, detect objects, track them across frames, and view or download the processed video.

## Features

- Upload video files in MP4, AVI, MOV, and MKV formats.
- Detect objects using a pre-trained YOLO11s model.
- Draw bounding boxes around detected objects.
- Display object labels and tracking IDs using Deep SORT.
- Process video frames with configurable confidence and inference settings.
- View the processed video directly in the Streamlit application.
- Download the processed video.

## Technologies Used

- **Python** — Application programming
- **OpenCV** — Video reading and frame processing
- **YOLO11s (Ultralytics)** — Object detection
- **Deep SORT** — Object tracking
- **Streamlit** — Web interface
- **PyTorch** — Deep learning framework
- **FFmpeg** — Video conversion for browser compatibility

## Project Structure

```text
Object-Detection-and-Tracking/
├── app.py
├── README.md
└── requirements.txt
```

*Replace `app.py` with your actual Python filename if it is different.*

## Installation

### 1. Clone or download the project

Open the project folder in Visual Studio Code.

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install streamlit opencv-python ultralytics deep-sort-realtime imageio-ffmpeg torch
```

If PyTorch installation fails, install the compatible version for your system using the official PyTorch installation instructions.

## How to Run

Open the terminal in the project folder and execute:

```bash
streamlit run app.py
```

Streamlit will provide a local URL to open the application in your browser.

## How to Use

1. Open the application.
2. Upload a supported video file.
3. Click **Start Detection and Tracking**.
4. Wait for the video to finish processing.
5. View the processed video in the application.
6. Download the output video using the download button.

## How It Works

1. **Video input:** OpenCV reads the uploaded video and extracts frames.
2. **Object detection:** YOLO identifies objects in the frames and generates bounding boxes and class labels.
3. **Object tracking:** Deep SORT associates detections across frames and assigns tracking IDs.
4. **Visualization:** Bounding boxes, labels, and tracking IDs are drawn onto the video frames.
5. **Output:** The processed frames are saved as a video, converted into a browser-compatible format when necessary, and displayed in Streamlit.

## Configuration

The application uses the following detection and processing settings:

- Confidence threshold: `0.45`
- Inference image size: `480`
- Frame skip: `2`
- Maximum processing dimension: `640`

These settings can be adjusted in the Python source code when required.

## Notes

- This implementation supports uploaded video files; webcam input is not currently included.
- The YOLO model weights may be downloaded automatically when the model is first loaded.
- Processing time depends on video length, resolution, and available hardware.
- Detection accuracy depends on the model, video quality, lighting, and objects present in the footage.

## Conclusion

This project demonstrates object detection and multi-object tracking in video using a pre-trained YOLO model and Deep SORT, with a Streamlit interface for uploading videos, viewing results, and downloading processed output.
