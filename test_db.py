"""
Test suite for JewelleryDBManager.

Run:
    python test_db.py

Expected output:
    ALL TESTS PASSED
"""

from datetime import datetime, timedelta
from database.db_manager import JewelleryDBManager


def test_all():
    # Use in-memory DB for testing
    db = JewelleryDBManager(db_path=":memory:")

    # ---- Record 1: Normal detection ----
    t1 = datetime(2024, 3, 15, 14, 30, 22)
    row1 = db.insert_detection(
        video_path="runs/recordings/gold_detected_20240315_143022.mp4",
        weight="12.5g",
        captured_at=t1
    )
    assert row1 is not None, "Record 1 should be inserted"

    # ---- Record 2: Unknown weight ----
    t2 = datetime(2024, 3, 15, 14, 35, 0)
    row2 = db.insert_detection(
        video_path="runs/recordings/gold_detected_20240315_143500.mp4",
        weight="None",
        captured_at=t2
    )
    assert row2 is not None, "Record 2 (unknown weight) should be inserted"

    # Verify it stored 'unknown'
    rec2 = db.get_detection_by_id(db.generate_unique_id(t2, "None"))
    assert rec2 is not None, "Record 2 should be found by unique_id"
    assert rec2["weight_grams"] == "unknown", f"Weight should be 'unknown', got '{rec2['weight_grams']}'"

    # ---- Record 3: Soft duplicate of Record 1 ----
    # Same weight ±0.3g, within 25 seconds → should be rejected
    t3 = t1 + timedelta(seconds=25)
    row3 = db.insert_detection(
        video_path="runs/recordings/gold_detected_20240315_143047.mp4",
        weight="12.8",
        captured_at=t3
    )
    assert row3 is None, "Record 3 should be rejected as soft duplicate (same weight ±0.5g within 30s)"

    # ---- Verify pending image extraction ----
    pending = db.get_pending_image_extraction()
    assert len(pending) == 2, f"Should have 2 pending records, got {len(pending)}"

    # All rows should have image_path = None
    for p in pending:
        assert p["image_path"] is None, "Pending records should have image_path = NULL"

    # ---- Verify update_image_path ----
    db.update_image_path(row1, "runs/images/gold_detected_20240315_143022_gold.jpg")
    updated = db.get_all_detections()
    for r in updated:
        if r["id"] == row1:
            assert r["image_path"] == "runs/images/gold_detected_20240315_143022_gold.jpg"
            assert r["image_extracted_at"] is not None

    # After update, only 1 pending should remain
    pending_after = db.get_pending_image_extraction()
    assert len(pending_after) == 1, f"Should have 1 pending after update, got {len(pending_after)}"

    # ---- Test exact duplicate rejection ----
    row1_dup = db.insert_detection(
        video_path="runs/recordings/gold_detected_20240315_143022.mp4",
        weight="12.5g",
        captured_at=t1
    )
    assert row1_dup is None, "Exact duplicate should be rejected"

    # ---- Test weight normalization ----
    assert JewelleryDBManager._normalize_weight("12.5g") == "12.5"
    assert JewelleryDBManager._normalize_weight("12.5 g") == "12.5"
    assert JewelleryDBManager._normalize_weight("12.5 grams") == "12.5"
    assert JewelleryDBManager._normalize_weight("None") == "unknown"
    assert JewelleryDBManager._normalize_weight("") == "unknown"
    assert JewelleryDBManager._normalize_weight("OCR error") == "unknown"

    # ---- Test re-placement after >60 seconds (should insert) ----
    t4 = t1 + timedelta(seconds=65)
    row4 = db.insert_detection(
        video_path="runs/recordings/gold_detected_20240315_143127.mp4",
        weight="12.5",
        captured_at=t4
    )
    assert row4 is not None, "Detection after >60s should be treated as new"

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    test_all()
