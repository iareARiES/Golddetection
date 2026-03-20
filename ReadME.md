# Gold Detection and Segmentation System

Real-time gold/jewellery detection system using YOLO models with person segmentation to suppress false positives, OCR weight reading, auto-recording, and a SQLite database for logging every detection event.

## How It Works

1. **Gold Detection** — Custom YOLO11n model detects gold objects in a defined ROI
2. **Person Segmentation** — YOLO26-seg masks out people so gold worn by a person is ignored
3. **Weight Reading** — EasyOCR reads the weight from the scale display
4. **Auto Recording** — Video is recorded automatically when gold is detected, stops 10s after last detection
5. **Database Logging** — Every detection is logged to SQLite with video path, weight, timestamp, and extracted image

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the detection system
python3 GoldNormal.py
```

Press **Q** to quit.

## Requirements

### Python Dependencies

```bash
pip install ultralytics opencv-python easyocr numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Optional: Database Viewer

To visually browse the detection database:

```bash
sudo apt install sqlitebrowser
sqlitebrowser runs/jewellery_detections.db
```

## Project Structure

```
Golddetection/
├── GoldNormal.py                  ← Main script (run this)
├── database/                      ← Database management package
│   ├── __init__.py
│   ├── db_manager.py              ← SQLite DB with deduplication
│   ├── image_extractor.py         ← Extracts best gold frame from videos
│   └── post_processor.py          ← Background thread for image extraction
├── weights/                       ← All YOLO model files
│   ├── Yolo11n.engine             ← Gold detection (TensorRT)
│   ├── yolo26n-seg.onnx           ← Person segmentation (ONNX)
│   └── ...                        ← Other model formats (.pt, .onnx)
├── runs/
│   ├── recordings/                ← Auto-saved .mp4 clips
│   ├── images/                    ← Snapshots and extracted gold frames
│   ├── jewellery_detections.db    ← SQLite database (auto-created)
│   └── detection.log              ← Event log
├── test_db.py                     ← Database test suite
├── data.yaml                      ← Dataset config (Roboflow)
├── requirements.txt
└── ReadME.md
```

## Models

| Model | Format | Purpose |
|---|---|---|
| `Yolo11n.engine` | TensorRT | Gold/jewellery detection (primary) |
| `yolo26n-seg.onnx` | ONNX | Person segmentation |

All models are stored in the `weights/` directory.

### Detection Classes (from `data.yaml`)

Bangles · Chain · Earrings · Gold Bar · Gold Coin · Ring

## Database

Detection events are stored in `runs/jewellery_detections.db` (SQLite, auto-created on first run).

### Checking the Database

**Command line:**
```bash
sqlite3 -header -column runs/jewellery_detections.db "SELECT * FROM gold_detections;"
```

**GUI viewer:**
```bash
sudo apt install sqlitebrowser
sqlitebrowser runs/jewellery_detections.db
```

**Python:**
```python
from database.db_manager import JewelleryDBManager
db = JewelleryDBManager()
for row in db.get_all_detections():
    print(row)
```

### Running Tests

```bash
python3 test_db.py
# Expected output: ALL TESTS PASSED
```

## Dependencies

- [PyTorch](https://pytorch.org/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [OpenCV](https://opencv.org/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [NumPy](https://numpy.org/)
- SQLite3 (built-in with Python)
