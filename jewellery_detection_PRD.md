# PRD: Jewellery Detection System — Database Management Module

**Project:** Gold Detection System — Database Layer Extension  
**Target IDE:** Anthropic Claude Opus (Antigravity IDE)  
**Document Type:** Product Requirements Document (Prompt/Spec for Coding Agent)  
**Version:** 1.0  

---

## 🧠 Context for the Coding Agent

You are extending an existing Python computer vision pipeline that:

1. **Detects gold objects** on a weighing machine using a custom YOLO11n model (`.engine` format)
2. **Segments humans** using YOLOv8-seg (`.onnx`) to suppress false positives — gold worn by a person is ignored
3. **Reads weight** from a scale display using EasyOCR
4. **Records video** automatically when gold is detected, saving `.mp4` clips to `runs/recordings/`

The existing codebase is provided below. You are **NOT** rewriting the detection logic — only adding a **database management layer** on top of it.

---

## 📎 Existing Codebase Reference

```
MultiDetectorROI.run()          ← Main loop
GoldDetectorROI                 ← Gold detection + recording
YOLOSegmentation                ← Person segmentation
MultiDetectorROI.run_ocr_on_roi() ← Weight reading via OCR
runs/recordings/                ← Where .mp4 files are currently saved
```

Recording filenames follow this pattern:
```
gold_detected_YYYYMMDD_HHMMSS.mp4
```

---

## 🎯 Goal

Add a complete **database management system** that:

1. During **live inferencing** — logs each gold detection event (video saved, weight, timestamp) into a SQLite database
2. After detection — runs a **post-processing pipeline** on saved videos to extract a representative **still image** of the gold piece
3. Exposes a **query interface** for retrieving records

---

## 📐 Database Schema

Create a SQLite database at `runs/jewellery_detections.db`.

### Table: `gold_detections`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | Internal row ID |
| `unique_id` | TEXT UNIQUE | Deduplication key: `{YYYYMMDD}_{HHMMSS}_{weight_grams}` e.g. `20240315_143022_12.5g` |
| `video_path` | TEXT | Absolute or relative path to the saved `.mp4` file |
| `image_path` | TEXT | Path to extracted still image (`.jpg`), NULL until post-processing runs |
| `weight_grams` | TEXT | OCR-read weight as string (e.g. `"12.5"`, `"None"` if unreadable) |
| `captured_at` | DATETIME | ISO 8601 timestamp when recording started |
| `image_extracted_at` | DATETIME | Timestamp when post-processing ran, NULL until then |
| `is_duplicate` | INTEGER | 0 = unique, 1 = duplicate (skipped), default 0 |
| `notes` | TEXT | Optional field for manual annotation |

---

## 🧩 Module Breakdown

### Module 1: `database/db_manager.py`

**Class: `JewelleryDBManager`**

Responsibilities:
- Initialize SQLite DB and create table if not exists
- Insert new detection records
- Check for duplicates before inserting
- Update a record with the extracted image path after post-processing
- Provide query methods

**Methods to implement:**

```python
def __init__(self, db_path: str = "runs/jewellery_detections.db"):
    """Connect to SQLite DB, create table if not exists."""

def generate_unique_id(self, captured_at: datetime, weight: str) -> str:
    """
    Generate deduplication key.
    Format: YYYYMMDD_HHMMSS_{weight}g
    Example: 20240315_143022_12.5g
    If weight is 'None' or unreadable → use 'unknown' in the ID.
    """

def is_duplicate(self, unique_id: str) -> bool:
    """
    Check if a record with this unique_id already exists.
    Also perform SOFT duplicate check:
      - If a record exists within ±30 seconds AND same weight → treat as duplicate
      - This handles cases where weight reading is slightly different due to OCR noise
    Returns True if duplicate.
    """

def insert_detection(self, video_path: str, weight: str, captured_at: datetime) -> int | None:
    """
    Insert a new gold detection event.
    - Generate unique_id first
    - Check is_duplicate() → if True, log warning and return None
    - Insert row with image_path = NULL
    - Return the new row id
    """

def update_image_path(self, row_id: int, image_path: str):
    """Update the image_path and image_extracted_at for a given row."""

def get_all_detections(self) -> list[dict]:
    """Return all rows as list of dicts."""

def get_detection_by_id(self, unique_id: str) -> dict | None:
    """Lookup by unique_id."""

def get_pending_image_extraction(self) -> list[dict]:
    """Return all rows where image_path IS NULL and video_path exists."""
```

---

### Module 2: `database/image_extractor.py`

**Class: `GoldImageExtractor`**

Responsibilities:
- Given a `.mp4` video path, extract the **best representative frame** of the gold piece
- Save it as a `.jpg` in `runs/images/`
- Return the saved image path

**Frame selection strategy:**

The best frame is selected using this priority order:
1. Scan frames where gold detection confidence is highest (re-run YOLO gold model on sampled frames)
2. If YOLO re-inference is too slow, fall back to: take the frame at **30% into the video duration** (gold is usually most visible early in detection)
3. Skip blurry frames using **Laplacian variance** — if variance < 100, skip that frame

**Methods to implement:**

```python
def __init__(self, gold_model_path: str, output_dir: str = "runs/images/"):
    """Load gold model, prepare output dir."""

def compute_blur_score(self, frame: np.ndarray) -> float:
    """Return Laplacian variance. Higher = sharper."""

def extract_best_frame(self, video_path: str) -> str | None:
    """
    Open video, sample every Nth frame (N = max(1, total_frames // 20)).
    For each sampled frame:
      - Check blur score (skip if < 100)
      - Run gold YOLO inference
      - Track frame with highest detection confidence
    Save best frame as {video_stem}_gold.jpg in output_dir.
    If no valid frame found, fall back to frame at 30% of video.
    Return saved image path, or None on failure.
    """
```

---

### Module 3: `database/post_processor.py`

**Class: `PostProcessor`**

Responsibilities:
- Poll for videos that have been saved but not yet had images extracted
- Run `GoldImageExtractor` on each pending video
- Update the DB record with the resulting image path

**Design:**
- This runs as a **background thread** inside the main `MultiDetectorROI.run()` loop
- It checks for pending records every **15 seconds**
- It should not block the main detection thread

**Methods to implement:**

```python
def __init__(self, db_manager: JewelleryDBManager, extractor: GoldImageExtractor):
    ...

def process_pending(self):
    """
    Get all records where image_path IS NULL.
    For each, run extractor.extract_best_frame(video_path).
    If successful, call db_manager.update_image_path(row_id, image_path).
    Log success/failure.
    """

def start_background_thread(self):
    """
    Start a daemon thread that calls process_pending() every 15 seconds.
    Use threading.Thread with daemon=True.
    """
```

---

### Module 4: Integration into `MultiDetectorROI`

Modify the existing `MultiDetectorROI` class to integrate the database layer.

**Changes required in `__init__`:**

```python
from database.db_manager import JewelleryDBManager
from database.image_extractor import GoldImageExtractor
from database.post_processor import PostProcessor

# Add to __init__:
self.db = JewelleryDBManager()
self.extractor = GoldImageExtractor(gold_model_path=gold_model_path)
self.post_processor = PostProcessor(self.db, self.extractor)
self.post_processor.start_background_thread()
self._current_db_row_id = None  # Track current recording's DB row
```

**Changes required in `run()` loop:**

After `self.gold_detector.handle_recording(frame, gold_detected)`:

```python
# When recording STARTS (was_recording=False → now True):
if self.gold_detector.recording and not was_recording:
    row_id = self.db.insert_detection(
        video_path=str(self.gold_detector.out_file),
        weight=self.last_ocr_text,
        captured_at=datetime.now()
    )
    self._current_db_row_id = row_id
```

**Also capture a photo at detection start:**

When recording starts, immediately save a snapshot photo (in addition to video):

```python
# Save snapshot photo alongside video
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
snapshot_path = Path("runs/images") / f"snapshot_{ts}.jpg"
snapshot_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(snapshot_path), frame)
```

---

## 🔁 Deduplication Logic (Edge Case Handling)

The system must prevent saving the same gold piece multiple times.

### Rule 1 — Exact Duplicate (same unique_id)
If `unique_id` already exists in DB → skip insertion entirely, log: `"Duplicate skipped: {unique_id}"`

### Rule 2 — Soft Duplicate (same gold piece, slight OCR variation)
If a record exists in DB where:
- `captured_at` is within ±30 seconds of the new detection
- AND `weight_grams` matches (with tolerance: `abs(float(new) - float(existing)) < 0.5`)

→ Treat as duplicate. Skip. Log: `"Soft duplicate detected — likely same gold piece"`

### Rule 3 — Unknown weight
If OCR returns `"None"` for weight:
- Still insert a record (gold IS detected)
- `weight_grams = "unknown"`
- `unique_id = {YYYYMMDD_HHMMSS}_unknown`
- Do NOT apply soft duplicate check on weight for these records

### Rule 4 — Re-placement of same gold piece
If the same piece is removed and placed back after **>60 seconds** → treat as a NEW detection (insert new record). The 30-second soft window handles brief jitter, not re-measurements.

---

## 📸 Photo Capture Logic

Two types of images are captured per detection:

| Type | When | How | Saved To |
|---|---|---|---|
| **Snapshot** | Immediately when recording starts | `cv2.imwrite()` of current frame | `runs/images/snapshot_{timestamp}.jpg` |
| **Extracted Best Frame** | Post-processing (background thread) | Re-inference on saved video | `runs/images/{video_stem}_gold.jpg` |

Both paths are stored in the DB. The `image_path` column holds the **extracted best frame** path (higher quality). The snapshot path can be stored in `notes` column temporarily until best frame extraction completes, or in an optional `snapshot_path` column (your choice — document whichever you implement).

---

## 📁 Directory Structure

```
project_root/
├── main.py                          ← Existing file (modify in-place)
├── database/
│   ├── __init__.py
│   ├── db_manager.py                ← NEW
│   ├── image_extractor.py           ← NEW
│   └── post_processor.py           ← NEW
├── runs/
│   ├── recordings/                  ← Existing: .mp4 files saved here
│   ├── images/                      ← NEW: .jpg extractions saved here
│   └── jewellery_detections.db      ← NEW: SQLite database
```

---

## ⚙️ Dependencies

All dependencies are already available in the project environment except:

```bash
# Already available:
# - opencv-python (cv2)
# - ultralytics (YOLO)
# - easyocr
# - numpy

# Standard library (no install needed):
# - sqlite3
# - threading
# - pathlib
# - datetime
# - logging
```

No new pip installs required. Use only `sqlite3` (built-in) for the database.

---

## 🪵 Logging

Use Python's `logging` module (not `print`) for all database and post-processing events.

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("runs/detection.log"),
        logging.StreamHandler()
    ]
)
```

Log these events at minimum:
- `INFO` — New detection inserted: `{unique_id}`
- `INFO` — Duplicate skipped: `{unique_id}`
- `INFO` — Image extraction complete: `{image_path}`
- `WARNING` — OCR weight unreadable, using 'unknown'
- `ERROR` — Video not found for post-processing: `{video_path}`
- `ERROR` — Frame extraction failed: `{error}`

---

## 🧪 Testing Requirements

Write a `test_db.py` script at the project root that:

1. Creates an in-memory SQLite DB (`db_path=":memory:"`)
2. Inserts 3 test records (one with `weight="None"`, two with same weight ±0.3g within 25 seconds)
3. Asserts the second near-duplicate is correctly rejected
4. Asserts the unknown-weight record is inserted
5. Asserts `get_pending_image_extraction()` returns the correct rows
6. Prints `ALL TESTS PASSED` if successful

---

## ✅ Acceptance Criteria

The implementation is complete when:

- [ ] Running `main.py` creates `runs/jewellery_detections.db` automatically on first launch
- [ ] Every time a gold recording starts, a new row appears in the DB within 1 second
- [ ] Snapshots are saved to `runs/images/` at recording start
- [ ] Background thread runs post-processing without blocking live detection
- [ ] Extracted best-frame images are saved and linked to DB rows within ~30 seconds of recording end
- [ ] Soft duplicate detection prevents double-logging the same gold piece
- [ ] `test_db.py` runs and prints `ALL TESTS PASSED`
- [ ] `runs/detection.log` contains structured logs of all events

---

## 🚫 Out of Scope

- No web UI or REST API required
- No cloud database (SQLite only)
- No changes to the YOLO model or OCR logic
- No multi-camera support
- No alert/notification system

---

## 💬 Notes for Claude Opus

- Preserve all existing class interfaces in `main.py` — do not rename methods or change constructor signatures
- The `GoldDetectorROI.out_file` attribute holds the current recording's `Path` object — use this to get the video path
- Thread safety: the background `PostProcessor` thread only reads `db.get_pending_image_extraction()` and writes `db.update_image_path()` — the main thread only calls `db.insert_detection()`. These are non-conflicting operations but add a `threading.Lock()` to `JewelleryDBManager` for safety
- OCR weight values may contain units (e.g. `"12.5g"`, `"12.5 g"`, `"12.5"`) — normalize to float string without units before storing
- When video path is not yet finalized at recording start (file may still be open), store the path anyway — the post-processor checks file existence before opening

---

*End of PRD — Ready for Claude Opus implementation*
