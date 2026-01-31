# Gold Detection and Segmentation

This project contains Python scripts for detecting gold and segmenting objects in a video stream using YOLO (You Only Look Once) models.

## Scripts

### `detect.py`

This script uses a YOLO model to detect gold in a webcam feed. It focuses on a specific region of interest (ROI), draws bounding boxes around detected gold, and records a video when gold is detected.

**Usage:**

```bash
python detect.py
```

### `GoldNormal.py`

This script performs segmentation on a webcam feed using a YOLO model and displays the annotated frame.

**Usage:**

```bash
python GoldNormal.py
```

### `GoldSegmentation.py`

This is a more advanced script that combines gold detection with person segmentation. It uses two YOLO models:

1.  A model for gold detection.
2.  A model for person segmentation.

The script detects gold only if it does not overlap with a person in the defined region of interest (ROI). This is useful for preventing false positives where a person might be wearing gold.

**Usage:**

```bash
python GoldSegmentation.py
```

## Models

The following YOLO models are used in this project:

*   `best.pt`: A model trained for gold detection.
*   `yolo26n-seg.pt`: A model for segmentation.
*   `Gold.pt`: (Not explicitly used in the scripts, but likely another gold detection model)
*   `yoloe-26n-seg.pt`: (Not explicitly used in the scripts, but likely another segmentation model)

## Data

The `data.yaml` file defines the dataset configuration for training the gold detection model. The dataset was created using Roboflow and contains the following classes:

*   Bangles
*   Chain
*   Earrings
*   Gold Bar
*   Gold Coin
*   Ring

## Dependencies

*   [PyTorch](https://pytorch.org/)
*   [Ultralytics YOLO](https://docs.ultralytics.com/)
*   [OpenCV](https://opencv.org/)

You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```
