import sqlite3
from pathlib import Path
from datetime import date


class DetectionReader:
    """Read-only interface to the jewellery detections database."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Resolve relative to project root, not CWD
            db_path = str(Path(__file__).parent.parent / "runs" / "jewellery_detections.db")
        self.db_path = db_path

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _db_exists(self) -> bool:
        return Path(self.db_path).exists()

    def get_all(self, filter_mode: str = "all") -> list[dict]:
        """
        filter_mode options:
          "all"         → all rows, order by captured_at DESC
          "pending"     → rows where image_path IS NULL
          "today"       → rows where date(captured_at) = today
          "with_weight" → rows where weight_grams != 'unknown'
          "duplicates"  → rows where is_duplicate = 1
        """
        if not self._db_exists():
            return []

        base = "SELECT * FROM gold_detections"
        filters = {
            "all": "",
            "pending": " WHERE image_path IS NULL AND is_duplicate = 0",
            "today": f" WHERE date(captured_at) = '{date.today().isoformat()}'",
            "with_weight": " WHERE weight_grams != 'unknown' AND is_duplicate = 0",
            "duplicates": " WHERE is_duplicate = 1",
        }
        where = filters.get(filter_mode, "")
        query = f"{base}{where} ORDER BY captured_at DESC"

        conn = self._get_connection()
        try:
            rows = conn.execute(query).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def search(self, query: str) -> list[dict]:
        """
        Full text search across: unique_id, weight_grams, notes.
        Uses SQL LIKE with % wildcards.
        """
        if not self._db_exists() or not query.strip():
            return []

        pattern = f"%{query.strip()}%"
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """SELECT * FROM gold_detections
                   WHERE unique_id LIKE ?
                      OR weight_grams LIKE ?
                      OR notes LIKE ?
                   ORDER BY captured_at DESC""",
                (pattern, pattern, pattern)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Return counts: total, pending_image, duplicates, today_count."""
        if not self._db_exists():
            return {"total": 0, "pending_image": 0, "duplicates": 0, "today_count": 0}

        conn = self._get_connection()
        try:
            total = conn.execute("SELECT COUNT(*) FROM gold_detections").fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM gold_detections WHERE image_path IS NULL AND is_duplicate = 0"
            ).fetchone()[0]
            dupes = conn.execute(
                "SELECT COUNT(*) FROM gold_detections WHERE is_duplicate = 1"
            ).fetchone()[0]
            today = conn.execute(
                f"SELECT COUNT(*) FROM gold_detections WHERE date(captured_at) = '{date.today().isoformat()}'"
            ).fetchone()[0]
            return {
                "total": total,
                "pending_image": pending,
                "duplicates": dupes,
                "today_count": today,
            }
        finally:
            conn.close()

    def get_by_id(self, row_id: int) -> dict | None:
        """Fetch a single record by primary key id."""
        if not self._db_exists():
            return None

        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM gold_detections WHERE id = ?", (row_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
