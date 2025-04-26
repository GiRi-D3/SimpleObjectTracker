# Human Brain Inspired Predictive and Attention-Based Object Tracking Model

**Author**: Dwaipayan Giri

## Project Overview

The **Human Brain Inspired Predictive and Attention-Based Object Tracking Model** aims to mimic the human visual system (HVS) for efficient object tracking in video sequences. Inspired by how the brain processes visual inputs with attention mechanisms and predictive tracking, this model seeks to reduce computational costs by focusing only on relevant areas of the frame, as opposed to traditional exhaustive frame scanning.

### Key Features
- **Predictive Motion Estimation**: Predicts future object locations based on past positions, minimizing unnecessary computations.
- **Attention Mechanism**: Focuses on the predicted region, expanding only when necessary.
- **Camera Motion Awareness**: Detects camera movement (e.g., panning, zooming) to adjust object tracking accordingly.
- **Handling Occlusions**: Capable of tracking objects even when they are partially or completely occluded.
- **Real-time Tracking**: The system is designed to function efficiently in real-time applications.

## Input and Output

### Input:
- A video file where the user selects an object to track.
  - **Manual Selection**: The user draws a bounding box around the object.
  - **Automatic Selection**: Objects are detected in the initial frames and automatically selected for tracking.

### Output:
- A sequence of bounding boxes or segmented pixels identifying the tracked object across frames, including handling motion, occlusion, and camera movement.

## Core Components

### A. **Predictive Tracking System**
   - **Purpose**: Predicts the future position of the object based on motion history.
   - **Approach**: Combines Kalman filters for motion prediction and ConvLSTM for capturing sequential motion patterns. Focuses on the predicted area and expands only when necessary.

### B. **Object Localization System**
   - **Purpose**: Detects the object in each frame, within the predicted region or by scanning the entire frame if necessary.
   - **Approach**: Uses YOLO for object detection and Siamese networks to track object re-appearance after occlusion.

## Datasets

The model is trained and evaluated using the following benchmark dataset:
- **TrackingNet**

## Goals

- **Efficiency**: Minimize computational costs by focusing only on relevant areas.
- **Accuracy**: Maintain high tracking accuracy even in the face of motion, occlusion, or camera movement.
- **Resilience**: Ensure robustness in real-time tracking, preventing failures in cases where the object goes out of frame or undergoes transformations such as changes in shape or color.

## Getting TrackingNet Dataset

Follow the steps below to download the TrackingNet dataset:

1. Install the TrackingNet pip package:
    ```bash
    pip install TrackingNet
    ```

2. Create an access token on Hugging Face by visiting [this link](https://huggingface.co/settings/tokens). You will need to create an account or log in if you already have one.

3. Paste the token in `download_trackingnet.py` and run it:
    ```bash
    python download_trackingnet.py
    ```
4. This will create a folder called `TrackingNet` and store the dataset inside it.

5. Then run `extract_trackingnet.py`:
   ```bash
   python extract_trackingnet.py
   ```
