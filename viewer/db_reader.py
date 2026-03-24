import sqlite3
from pathlib import Path
from datetime import date


class DetectionReader:
    """Read-only interface to runs/gold.db (new schema with processing_status)."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "runs" / "gold.db")
        self.db_path = db_path

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _exists(self) -> bool:
        return Path(self.db_path).exists()

    def get_all(self, filter_mode: str = "all") -> list[dict]:
        if not self._exists():
            return []

        base = "SELECT * FROM detections"
        filters = {
            "all":         "",
            "today":       f" WHERE date(captured_at) = '{date.today().isoformat()}'",
            "with_weight": " WHERE weight NOT IN ('None', 'unavailable', '') AND weight IS NOT NULL",
            "pending":     " WHERE processing_status IN ('pending', 'processing')",
            "partial":     " WHERE processing_status = 'partial'",
            "failed":      " WHERE processing_status = 'failed'",
            "done":        " WHERE processing_status = 'done'",
        }
        where = filters.get(filter_mode, "")
        query = f"{base}{where} ORDER BY captured_at DESC"

        conn = self._conn()
        try:
            return [dict(r) for r in conn.execute(query).fetchall()]
        finally:
            conn.close()

    def search(self, query: str) -> list[dict]:
        if not self._exists() or not query.strip():
            return []

        pattern = f"%{query.strip()}%"
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT * FROM detections
                   WHERE event_id LIKE ?
                      OR weight LIKE ?
                      OR c270_video_path LIKE ?
                      OR captured_at LIKE ?
                      OR processing_status LIKE ?
                   ORDER BY captured_at DESC""",
                (pattern, pattern, pattern, pattern, pattern),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_stats(self) -> dict:
        if not self._exists():
            return {"total": 0, "pending": 0, "done": 0, "failed": 0, "today_count": 0}

        conn = self._conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE processing_status IN ('pending','processing')"
            ).fetchone()[0]
            done = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE processing_status='done'"
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE processing_status='failed'"
            ).fetchone()[0]
            today = conn.execute(
                f"SELECT COUNT(*) FROM detections WHERE date(captured_at) = '{date.today().isoformat()}'"
            ).fetchone()[0]
            return {"total": total, "pending": pending, "done": done,
                    "failed": failed, "today_count": today}
        finally:
            conn.close()

    def get_by_id(self, row_id: int) -> dict | None:
        if not self._exists():
            return None
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM detections WHERE id = ?", (row_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_by_event_id(self, event_id: str) -> dict | None:
        if not self._exists():
            return None
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM detections WHERE event_id = ?", (event_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def reset_to_pending(self, event_id: str):
        """Mark a failed row as pending so the viewer can re-enqueue it."""
        if not self._exists():
            return
        conn = self._conn()
        try:
            conn.execute(
                """UPDATE detections SET processing_status='pending',
                   processing_error=NULL WHERE event_id=?""",
                (event_id,),
            )
            conn.commit()
        finally:
            conn.close()
