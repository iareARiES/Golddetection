"""
GoldNormal.py -- Dual-camera gold detection system (v2)
========================================================

Architecture:  record first, process second.

Main thread (C270):
  detect gold -> start dual recording -> stop after 10s tail ->
  insert raw event as 'pending' -> enqueue for background processing

Lenovo thread:
  continuous read + rotate 180 -> frame buffer under lock

PostProcessWorker (single daemon thread, FIFO):
  pick pending event -> extract gold crop from C270 video ->
  match Lenovo frame -> run OCR on clean frame -> update DB row

Both cameras are mounted upside-down: rotate 180 before anything.
"""

import cv2
import time
import uuid
import queue
import threading
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

import json as _json
import numpy as np
import easyocr
from ultralytics import YOLO


# ---------------------------------------------------------------------------
#  LOGGING
# ---------------------------------------------------------------------------
Path("runs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("runs/detection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("GoldNormal")

# Project root (absolute) -- all paths relative to this
PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
#  DATABASE
# ---------------------------------------------------------------------------
class DB:
    """Thread-safe SQLite wrapper with pending/done/failed workflow."""

    def __init__(self, path="runs/gold.db"):
        abs_path = str((PROJECT_ROOT / path).resolve())
        (PROJECT_ROOT / path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(abs_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init()

    def _init(self):
        with self._lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id            TEXT UNIQUE,
                    captured_at         TEXT,
                    duration_sec        REAL,

                    c270_video_path     TEXT NOT NULL,
                    lenovo_video_path   TEXT,

                    weight              TEXT,
                    image_c270          TEXT,
                    image_lenovo        TEXT,

                    detection_confidence REAL,
                    bbox_json           TEXT,
                    sync_offset_ms      INTEGER,

                    processing_status   TEXT NOT NULL DEFAULT 'pending',
                    processing_error    TEXT,

                    queued_at           TEXT,
                    processed_at        TEXT
                )
            """)
            self.conn.commit()

    def insert_raw_event(self, event_id, captured_at, duration_sec,
                         c270_video_path, lenovo_video_path):
        """Insert immediately after both writers are released."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO detections
                       (event_id, captured_at, duration_sec,
                        c270_video_path, lenovo_video_path,
                        processing_status, queued_at)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                (event_id, captured_at, duration_sec,
                 c270_video_path, lenovo_video_path, now),
            )
            self.conn.commit()
            return cur.lastrowid

    def update_processed(self, event_id, weight, image_c270, image_lenovo,
                         detection_confidence=None, bbox_json=None,
                         sync_offset_ms=None):
        """Called by PostProcessWorker after extraction.
        Status is 'done' if all data present, 'partial' if some missing."""
        now = datetime.now().isoformat(timespec="seconds")
        # Determine status: partial if any key field is missing
        if image_c270 and weight and weight != "None":
            status = "done"
        else:
            status = "partial"
        with self._lock:
            self.conn.execute(
                """UPDATE detections
                   SET weight=?, image_c270=?, image_lenovo=?,
                       detection_confidence=?, bbox_json=?,
                       sync_offset_ms=?,
                       processing_status=?, processed_at=?
                   WHERE event_id=?""",
                (weight, image_c270, image_lenovo,
                 detection_confidence, bbox_json, sync_offset_ms,
                 status, now, event_id),
            )
            self.conn.commit()

    def update_failed(self, event_id, error_msg):
        """Called by PostProcessWorker on failure."""
        with self._lock:
            self.conn.execute(
                """UPDATE detections
                   SET processing_status='failed', processing_error=?
                   WHERE event_id=?""",
                (error_msg, event_id),
            )
            self.conn.commit()

    def update_processing(self, event_id):
        """Mark row as currently being processed."""
        with self._lock:
            self.conn.execute(
                """UPDATE detections SET processing_status='processing'
                   WHERE event_id=?""",
                (event_id,),
            )
            self.conn.commit()

    def get_pending_events(self):
        """Return event_ids of rows still pending or failed (for recovery)."""
        with self._lock:
            rows = self.conn.execute(
                """SELECT event_id FROM detections
                   WHERE processing_status IN ('pending', 'failed')
                   ORDER BY queued_at ASC"""
            ).fetchall()
            return [r["event_id"] for r in rows]

    def get_event(self, event_id):
        """Return a single event row as dict."""
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM detections WHERE event_id=?", (event_id,)
            ).fetchone()
            return dict(row) if row else None


# ---------------------------------------------------------------------------
#  LENOVO CAMERA  (daemon thread, passive frame buffer)
# ---------------------------------------------------------------------------
class LenovoCamera:
    """
    Runs in a daemon thread. Main loop never waits for it.
    capture_latest() returns the most recent rotated frame instantly.
    """

    def __init__(self, index: int = 2):
        self.cap = cv2.VideoCapture(index)
        self._available = self.cap.isOpened()
        if not self._available:
            log.warning("Lenovo cam (index=%d) unavailable.", index)
        self._frame = None
        self._lock  = threading.Lock()
        self._running = False

    @property
    def available(self):
        return self._available

    def start(self):
        if not self._available:
            return
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True, name="LenovoCam")
        t.start()
        log.info("Lenovo camera thread started.")

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.03)

    def capture_latest(self):
        """Return a copy of the latest frame, or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._available:
            self.cap.release()


# ---------------------------------------------------------------------------
#  PERSON SEGMENTATION  (YOLO)
# ---------------------------------------------------------------------------
class YOLOSegmentation:
    def __init__(self, model_path: str, roi: tuple, classes=None):
        self.model   = YOLO(model_path)
        self.roi     = roi
        self.classes = classes or [0]

    def _crop(self, frame):
        x1, y1, x2, y2 = self.roi
        return frame[y1:y2, x1:x2]

    def run(self, frame):
        """Return raw results on the ROI crop."""
        return self.model(self._crop(frame), classes=self.classes, conf=0.2)[0]

    def draw(self, frame, results):
        """Overlay masks + boxes on display_frame (with ROI offset)."""
        rx, ry = self.roi[0], self.roi[1]
        if results.masks is not None:
            for pts in results.masks.xy:
                c = np.array(pts, dtype=np.int32)
                c[:, 0] += rx;  c[:, 1] += ry
                overlay = frame.copy()
                cv2.fillPoly(overlay, [c], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.polylines(frame, [c], True, (0, 255, 0), 2)
        if results.boxes is not None:
            for box in results.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                bx1 += rx; bx2 += rx; by1 += ry; by2 += ry
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        return frame


# ---------------------------------------------------------------------------
#  GOLD DETECTOR
# ---------------------------------------------------------------------------
def _overlaps_person(box_coords, person_masks, roi):
    """Pixel-level check: True if gold box overlaps any person mask."""
    if not person_masks:
        return False
    x1, y1, x2, y2 = box_coords
    rx1, ry1, rx2, ry2 = roi
    w, h = rx2 - rx1, ry2 - ry1

    person_bin = np.zeros((h, w), dtype=np.uint8)
    for pts in person_masks:
        cv2.fillPoly(person_bin, [np.array(pts, dtype=np.int32)], 255)

    box_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(
        box_mask,
        (max(x1 - rx1, 0),     max(y1 - ry1, 0)),
        (min(x2 - rx1, w - 1), min(y2 - ry1, h - 1)),
        255, -1,
    )
    return bool(np.any(cv2.bitwise_and(person_bin, box_mask)))


class GoldDetector:
    def __init__(self, model_path: str, roi: tuple):
        self.model = YOLO(model_path)
        self.roi   = roi

    def _full_coords(self, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        rx, ry = self.roi[0], self.roi[1]
        return x1 + rx, y1 + ry, x2 + rx, y2 + ry

    def detect(self, frame, person_masks):
        """
        Run inference on clean_frame ROI, filter person overlaps.
        Returns list of (x1,y1,x2,y2) in full-frame coords.
        Does NOT draw anything.
        """
        x1r, y1r, x2r, y2r = self.roi
        results = self.model(frame[y1r:y2r, x1r:x2r], conf=0.2)[0]

        valid = []
        for box in results.boxes:
            coords = self._full_coords(box)
            if _overlaps_person(coords, person_masks, self.roi):
                continue
            valid.append(coords)
        return valid

    @staticmethod
    def draw_boxes(frame, gold_list):
        """Draw gold boxes on a display_frame copy only."""
        for cx1, cy1, cx2, cy2 in gold_list:
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
        return frame


# ---------------------------------------------------------------------------
#  DUAL VIDEO RECORDER  (10-second tail, two synchronized writers)
# ---------------------------------------------------------------------------
class DualVideoRecorder:
    TAIL_SECS = 10

    def __init__(self, out_dir: str = "runs/recordings"):
        self.out_dir = (PROJECT_ROOT / out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # active event state
        self._c270_writer   = None
        self._lenovo_writer = None
        self._event_id      = None
        self._captured_at   = None
        self._t0            = 0.0
        self._last_gold_t   = 0.0
        self.recording      = False

    @property
    def event_id(self):
        return self._event_id

    @property
    def c270_path(self):
        return str(self.out_dir / f"{self._event_id}_c270.mp4") if self._event_id else None

    @property
    def lenovo_path(self):
        return str(self.out_dir / f"{self._event_id}_lenovo.mp4") if self._event_id else None

    def feed(self, c270_frame, lenovo_frame, gold_detected, gold_list=None):
        """
        Feed both camera frames each loop iteration.

        Returns
        -------
        recording     bool
        just_stopped  bool
        completed     dict | None   -- event info when recording just ended
        """
        now = time.time()

        if gold_detected:
            self._last_gold_t = now
            if not self.recording:
                self._start(c270_frame, lenovo_frame)

        just_stopped = False
        completed    = None

        if self.recording:
            # Draw green gold boxes on the recorded C270 video
            rec_frame = c270_frame.copy()
            if gold_list:
                for (x1, y1, x2, y2) in gold_list:
                    cv2.rectangle(rec_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self._c270_writer.write(rec_frame)
            if self._lenovo_writer is not None and lenovo_frame is not None:
                self._lenovo_writer.write(lenovo_frame)

            tail_expired = (not gold_detected and
                            (now - self._last_gold_t) > self.TAIL_SECS)
            if tail_expired:
                completed = self._stop()
                just_stopped = True

        return self.recording, just_stopped, completed

    def force_stop(self):
        """Call on shutdown. Returns completed event dict or None."""
        if self.recording:
            return self._stop()
        return None

    def _start(self, c270_frame, lenovo_frame):
        self._event_id   = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self._captured_at = datetime.now().isoformat(timespec="seconds")
        self._t0          = time.time()

        # C270 writer
        h, w = c270_frame.shape[:2]
        self._c270_writer = cv2.VideoWriter(
            self.c270_path, self._fourcc, 20.0, (w, h),
        )

        # Lenovo writer (if frame available)
        if lenovo_frame is not None:
            lh, lw = lenovo_frame.shape[:2]
            self._lenovo_writer = cv2.VideoWriter(
                self.lenovo_path, self._fourcc, 20.0, (lw, lh),
            )
        else:
            self._lenovo_writer = None

        self.recording = True
        log.info("REC started  event=%s", self._event_id)

    def _stop(self):
        duration = time.time() - self._t0

        if self._c270_writer:
            self._c270_writer.release()
            self._c270_writer = None
        lenovo_path = None
        if self._lenovo_writer:
            self._lenovo_writer.release()
            self._lenovo_writer = None
            lenovo_path = self.lenovo_path

        self.recording = False
        log.info("REC stopped  event=%s  dur=%.1fs", self._event_id, duration)

        completed = {
            "event_id":          self._event_id,
            "captured_at":       self._captured_at,
            "duration_sec":      round(duration, 1),
            "c270_video_path":   self.c270_path,
            "lenovo_video_path": lenovo_path,
        }
        self._event_id = None
        return completed

# ---------------------------------------------------------------------------
#  OCR  (weight reader)
# ---------------------------------------------------------------------------
class OCRReader:
    def __init__(self, roi: tuple):
        self.roi = roi
        log.info("Initialising EasyOCR...")
        try:
            self.reader = easyocr.Reader(["en"], gpu=True)
            log.info("EasyOCR ready.")
        except Exception as exc:
            log.warning("EasyOCR init failed: %s", exc)
            self.reader = None

    def read(self, frame) -> str:
        """Return first digit-containing string found in ROI, else 'None'."""
        if self.reader is None:
            return "None"
        x1, y1, x2, y2 = self.roi
        try:
            for _, text, _ in self.reader.readtext(frame[y1:y2, x1:x2]):
                if any(ch.isdigit() for ch in text):
                    return text
        except Exception as exc:
            log.warning("OCR error: %s", exc)
        return "None"


# ---------------------------------------------------------------------------
#  UTILITIES
# ---------------------------------------------------------------------------
def get_screen_resolution():
    try:
        import tkinter as tk
        r = tk.Tk(); r.withdraw()
        res = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy();  return res
    except Exception:
        pass
    try:
        import subprocess
        for line in subprocess.check_output(["xrandr"]).decode().splitlines():
            if "*" in line:
                w, h = map(int, line.split()[0].split("x"))
                return w, h
    except Exception:
        pass
    return None


def resize_fit(frame, max_w, max_h):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_LINEAR)


def draw_hud(frame, gold_detected, recording, rec_start,
             last_weight, save_flash):
    """Draw status panel on display_frame only."""
    h, w = frame.shape[:2]
    px   = w - 240
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put(text, y, color):
        cv2.putText(frame, text, (px, y), font, 0.6, color, 2)

    put("Gold: YES" if gold_detected else "Gold: NO",
        30,  (0, 255, 0) if gold_detected else (0, 0, 255))
    put("REC:  ON " if recording else "REC:  OFF",
        65,  (0, 255, 0) if recording else (0, 0, 255))

    if recording and rec_start:
        e = int(time.time() - rec_start)
        put(f"Dur:  {e // 60:02d}:{e % 60:02d}", 100, (255, 255, 255))
    else:
        put("Dur:  00:00", 100, (255, 255, 255))

    wt = last_weight if last_weight != "None" else "--"
    put(f"Wt:   {wt}", 135, (255, 255, 0))

    if save_flash:
        put("Event saved!", 170, (0, 255, 255))

    return frame


# ---------------------------------------------------------------------------
#  MAIN SYSTEM
# ---------------------------------------------------------------------------
class DualCameraSystem:
    """
    Orchestrates C270 (detection) + Lenovo (context) with
    dual recording. Snapshots and OCR captured inline.
    """

    FLASH_SECS = 2.0

    def __init__(
        self,
        gold_model_path: str,
        seg_model_path:  str,
        roi:             tuple,
        c270_index:      int = 0,
        lenovo_index:    int = 2,
    ):
        self.roi = roi

        # -- cameras --
        self.cap = cv2.VideoCapture(c270_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open C270 (index={c270_index})")
        log.info("C270 opened (index=%d).", c270_index)

        self.lenovo = LenovoCamera(lenovo_index)
        self.lenovo.start()

        # -- detection models (main thread) --
        self.seg  = YOLOSegmentation(seg_model_path, roi)
        self.gold = GoldDetector(gold_model_path, roi)
        self.ocr  = OCRReader(roi)

        # -- subsystems --
        self.recorder = DualVideoRecorder()
        self.db       = DB()

        # -- per-event state --
        self._rec_start      = None
        self._prev_recording = False
        self._save_flash_t   = 0.0
        self._last_ocr_t     = 0.0
        self._last_weight    = "None"
        self._stop           = False

        # Snapshot data captured when gold first appears
        self._pending          = None
        self._ocr_settle_start = 0.0
        self._ocr_settled      = False

        Path(PROJECT_ROOT / "runs" / "images").mkdir(parents=True, exist_ok=True)

    # -- inline helpers ---
    def _save_snapshot(self, frame, prefix, event_id, crop_box=None):
        """Save a frame (optionally cropped to gold box) and return absolute path."""
        path = str(PROJECT_ROOT / "runs" / "images" / f"{event_id}_{prefix}.jpg")
        if crop_box:
            x1, y1, x2, y2 = crop_box
            h, w = frame.shape[:2]
            px1, py1 = max(0, x1 - 20), max(0, y1 - 20)
            px2, py2 = min(w, x2 + 20), min(h, y2 + 20)
            cv2.imwrite(path, frame[py1:py2, px1:px2])
        else:
            cv2.imwrite(path, frame)
        return path

    def _union_box(self, gold_list):
        """Compute union bounding rectangle of all gold boxes."""
        if not gold_list:
            return None
        x1 = min(b[0] for b in gold_list)
        y1 = min(b[1] for b in gold_list)
        x2 = max(b[2] for b in gold_list)
        y2 = max(b[3] for b in gold_list)
        return (x1, y1, x2, y2)

    def _on_gold_first_seen(self, clean_frame, gold_list, event_id):
        """
        Called once when gold first appears. Captures snapshots + OCR
        using the already-loaded models. No background worker needed.
        """
        ts = datetime.now().isoformat(timespec="seconds")

        # C270 gold crop (union of all valid gold boxes)
        union = self._union_box(gold_list)
        img_c270 = self._save_snapshot(clean_frame, "c270_crop", event_id, crop_box=union)

        # Best confidence from gold_list
        # (gold_list only has coords; re-run on ROI to get confidence)
        best_conf = None
        bbox_dict = None
        if union:
            bbox_dict = {"x1": union[0], "y1": union[1],
                         "x2": union[2], "y2": union[3]}
            # Get confidence from last detect call
            x1r, y1r, x2r, y2r = self.roi
            results = self.gold.model(clean_frame[y1r:y2r, x1r:x2r], conf=0.2, verbose=False)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                best_conf = max(box.conf.item() for box in results.boxes)
                best_conf = round(best_conf, 4)

        # Lenovo frame
        lenovo_frm = self.lenovo.capture_latest()
        img_lenovo = self._save_snapshot(lenovo_frm, "lenovo_frame", event_id) \
            if lenovo_frm is not None else None

        # OCR weight -- initial read; will be updated after 5s settle
        weight = self.ocr.read(clean_frame)

        self._pending = {
            "captured_at":          ts,
            "weight":               weight,
            "image_c270":           img_c270,
            "image_lenovo":         img_lenovo,
            "detection_confidence": best_conf,
            "bbox_json":            _json.dumps(bbox_dict) if bbox_dict else None,
        }
        # Start 5-second OCR settle timer
        self._ocr_settle_start = time.time()
        self._ocr_settled      = False
        log.info("Snapshots saved  C270=%s  Lenovo=%s  weight=%s  conf=%s",
                 img_c270, img_lenovo, weight, best_conf)

    def _on_recording_done(self, completed):
        """
        Called after both writers are released.
        Writes the complete DB row in one shot.
        """
        if self._pending is None:
            log.warning("Recording done but no pending data -- writing raw row.")
            row_id = self.db.insert_raw_event(**completed)
            log.info("DB row #%d inserted (pending) event=%s", row_id, completed["event_id"])
            self._save_flash_t = time.time()
            return

        ev = self._pending

        # First insert the raw event
        row_id = self.db.insert_raw_event(**completed)

        # Determine status
        if ev["image_c270"] and ev["weight"] and ev["weight"] != "None":
            status = "done"
        else:
            status = "partial"

        # Then immediately update with the inline-captured data
        self.db.update_processed(
            completed["event_id"],
            weight=ev["weight"],
            image_c270=ev["image_c270"],
            image_lenovo=ev["image_lenovo"],
            detection_confidence=ev["detection_confidence"],
            bbox_json=ev["bbox_json"],
            sync_offset_ms=0,  # inline capture = same moment
        )

        log.info("DB row #%d written (%s)  event=%s  weight=%s",
                 row_id, status, completed["event_id"], ev["weight"])

        self._pending      = None
        self._save_flash_t = time.time()

    # -- main loop --
    def run(self):
        win = "Gold Detection"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        res = get_screen_resolution()
        disp_w, disp_h = res if res else (1280, 720)
        cv2.resizeWindow(win, disp_w, disp_h)

        log.info("Main loop running -- press Q to quit.")

        while not self._stop:
            ret, frame = self.cap.read()
            if not ret:
                log.warning("C270 read failed -- retrying...")
                time.sleep(0.03)
                continue

            # -- 1. rotate 180 --
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # -- 2. clean_frame = unannotated source of truth --
            clean_frame = frame.copy()

            # -- 3. person segmentation (runs on clean_frame) --
            seg_results  = self.seg.run(clean_frame)
            person_masks = seg_results.masks.xy if seg_results.masks else None

            # -- 4. gold detection (on clean_frame, returns coords only) --
            gold_list     = self.gold.detect(clean_frame, person_masks)
            gold_detected = len(gold_list) > 0

            # -- 5. get latest Lenovo frame --
            lenovo_frame = self.lenovo.capture_latest()

            # -- 6. dual recording (writes clean frames only) --
            recording, just_stopped, completed = self.recorder.feed(
                clean_frame, lenovo_frame, gold_detected, gold_list
            )

            # -- 7. handle state transitions --
            if recording and not self._prev_recording:
                # Gold just appeared -> first frame of recording
                self._rec_start = time.time()
                event_id = self.recorder.event_id
                self._on_gold_first_seen(clean_frame, gold_list, event_id)

            elif not recording and self._prev_recording:
                self._rec_start = None

            if just_stopped and completed:
                self._on_recording_done(completed)

            self._prev_recording = recording

            # -- 8. OCR update while gold visible (5-second settle) --
            now = time.time()
            if gold_detected and self._pending and not self._ocr_settled:
                if (now - self._last_ocr_t) >= 1.0:
                    self._last_ocr_t = now
                    w = self.ocr.read(clean_frame)
                    if w != "None":
                        self._last_weight = w
                        self._pending["weight"] = w

                # After 5 seconds, lock in the weight
                if (now - self._ocr_settle_start) >= 5.0:
                    self._ocr_settled = True
                    log.info("OCR settled: weight=%s", self._pending.get("weight"))

            # -- 9. display_frame = annotated copy for HUD --
            display_frame = clean_frame.copy()
            display_frame = self.seg.draw(display_frame, seg_results)
            display_frame = GoldDetector.draw_boxes(display_frame, gold_list)

            save_flash = (now - self._save_flash_t) < self.FLASH_SECS

            display_frame = draw_hud(
                display_frame, gold_detected, recording,
                self._rec_start, self._last_weight, save_flash,
            )

            cv2.imshow(win, resize_fit(display_frame, disp_w, disp_h))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # -- shutdown --
        log.info("Shutting down...")
        completed = self.recorder.force_stop()
        if completed:
            self._on_recording_done(completed)

        self.cap.release()
        self.lenovo.stop()
        cv2.destroyAllWindows()
        log.info("Done.")


# ---------------------------------------------------------------------------
#  ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import signal

    ROI = (200, 200, 800, 800)

    system = DualCameraSystem(
        gold_model_path = "weights/Yolo11n.engine",
        seg_model_path  = "weights/yolo26n-seg.onnx",
        roi             = ROI,
        c270_index      = 0,
        lenovo_index    = 2,
    )

    _stop_flag = False
    def _sigint_handler(signum, frame_arg):
        global _stop_flag
        if _stop_flag:
            raise SystemExit(1)
        log.info("Ctrl+C received -- shutting down cleanly...")
        _stop_flag = True
        system._stop = True
    signal.signal(signal.SIGINT, _sigint_handler)

    system.run()