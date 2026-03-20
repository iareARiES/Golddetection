import sqlite3
import threading
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class JewelleryDBManager:
    """SQLite database manager for gold detection events."""

    def __init__(self, db_path: str = "runs/jewellery_detections.db"):
        """Connect to SQLite DB, create table if not exists."""
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.lock = threading.Lock()
        # Single persistent connection, protected by self.lock
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        with self.lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS gold_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    unique_id TEXT UNIQUE,
                    video_path TEXT,
                    image_path TEXT,
                    weight_grams TEXT,
                    captured_at DATETIME,
                    image_extracted_at DATETIME,
                    is_duplicate INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)
            self.conn.commit()

    @staticmethod
    def _normalize_weight(weight: str) -> str:
        """
        Strip units like 'g', ' g', 'grams' from OCR weight string.
        Returns cleaned numeric string or 'unknown'.
        """
        if weight is None or weight.strip().lower() in ("none", "", "ocr unavailable", "ocr error"):
            return "unknown"
        # Remove common unit suffixes
        cleaned = re.sub(r'\s*(grams|gram|g)\s*$', '', weight.strip(), flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if not cleaned:
            return "unknown"
        return cleaned

    def generate_unique_id(self, captured_at: datetime, weight: str) -> str:
        """
        Generate deduplication key.
        Format: YYYYMMDD_HHMMSS_{weight}g
        Example: 20240315_143022_12.5g
        If weight is 'None' or unreadable -> use 'unknown' in the ID.
        """
        ts = captured_at.strftime("%Y%m%d_%H%M%S")
        norm = self._normalize_weight(weight)
        if norm == "unknown":
            return f"{ts}_unknown"
        return f"{ts}_{norm}g"

    def is_duplicate(self, unique_id: str, captured_at: datetime, weight: str) -> bool:
        """
        Check if a record with this unique_id already exists.
        Also perform SOFT duplicate check:
          - If a record exists within ±30 seconds AND same weight (±0.5g) -> duplicate
          - Skip soft check for 'unknown' weight records
        Returns True if duplicate.
        """
        with self.lock:
            # Rule 1: Exact duplicate
            row = self.conn.execute(
                "SELECT id FROM gold_detections WHERE unique_id = ?",
                (unique_id,)
            ).fetchone()
            if row:
                logger.info("Duplicate skipped: %s", unique_id)
                return True

            # Rule 2: Soft duplicate (skip for unknown weight)
            norm = self._normalize_weight(weight)
            if norm != "unknown":
                try:
                    new_weight_f = float(norm)
                except ValueError:
                    return False

                window_start = (captured_at - timedelta(seconds=30)).isoformat()
                window_end = (captured_at + timedelta(seconds=30)).isoformat()

                rows = self.conn.execute(
                    """SELECT weight_grams FROM gold_detections
                       WHERE captured_at BETWEEN ? AND ?
                         AND weight_grams != 'unknown'
                         AND is_duplicate = 0""",
                    (window_start, window_end)
                ).fetchall()

                for r in rows:
                    try:
                        existing_w = float(r["weight_grams"])
                        if abs(new_weight_f - existing_w) < 0.5:
                            logger.info("Soft duplicate detected — likely same gold piece")
                            return True
                    except (ValueError, TypeError):
                        continue

            return False

    def insert_detection(self, video_path: str, weight: str, captured_at: datetime) -> int | None:
        """
        Insert a new gold detection event.
        - Generate unique_id first
        - Check is_duplicate() -> if True, log warning and return None
        - Insert row with image_path = NULL
        - Return the new row id
        """
        norm_weight = self._normalize_weight(weight)
        unique_id = self.generate_unique_id(captured_at, weight)

        if norm_weight == "unknown":
            logger.warning("OCR weight unreadable, using 'unknown'")

        if self.is_duplicate(unique_id, captured_at, weight):
            return None

        with self.lock:
            try:
                cursor = self.conn.execute(
                    """INSERT INTO gold_detections
                       (unique_id, video_path, image_path, weight_grams, captured_at,
                        image_extracted_at, is_duplicate, notes)
                       VALUES (?, ?, NULL, ?, ?, NULL, 0, NULL)""",
                    (unique_id, video_path, norm_weight, captured_at.isoformat())
                )
                self.conn.commit()
                row_id = cursor.lastrowid
                logger.info("New detection inserted: %s (row %d)", unique_id, row_id)
                return row_id
            except sqlite3.IntegrityError:
                logger.info("Duplicate skipped (integrity): %s", unique_id)
                return None

    def update_image_path(self, row_id: int, image_path: str):
        """Update the image_path and image_extracted_at for a given row."""
        with self.lock:
            self.conn.execute(
                """UPDATE gold_detections
                   SET image_path = ?, image_extracted_at = ?
                   WHERE id = ?""",
                (image_path, datetime.now().isoformat(), row_id)
            )
            self.conn.commit()
            logger.info("Image path updated for row %d: %s", row_id, image_path)

    def get_all_detections(self) -> list[dict]:
        """Return all rows as list of dicts."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM gold_detections ORDER BY captured_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_detection_by_id(self, unique_id: str) -> dict | None:
        """Lookup by unique_id."""
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM gold_detections WHERE unique_id = ?",
                (unique_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_pending_image_extraction(self) -> list[dict]:
        """Return all rows where image_path IS NULL and video_path exists."""
        with self.lock:
            rows = self.conn.execute(
                """SELECT * FROM gold_detections
                   WHERE image_path IS NULL
                     AND video_path IS NOT NULL
                     AND is_duplicate = 0"""
            ).fetchall()
            return [dict(r) for r in rows]
