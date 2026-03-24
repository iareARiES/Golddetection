import cv2
import numpy as np
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class GoldImageExtractor:
    """Extract the best representative gold frame from a recorded video."""

    def __init__(self, gold_model_path: str, output_dir: str = "runs/images/"):
        """Load gold model, prepare output dir."""
        self.model = YOLO(gold_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _has_moov_atom(video_path: str) -> bool:
        """
        Quick binary scan for 'moov' box in an MP4 file.
        Files without a moov atom are incomplete/corrupt and will crash
        FFMPEG at the native level (uncatchable from Python).
        """
        try:
            with open(video_path, 'rb') as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    if b'moov' in chunk:
                        return True
            return False
        except Exception:
            return False

    def compute_blur_score(self, frame: np.ndarray) -> float:
        """Return Laplacian variance. Higher = sharper."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

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
        video_path = str(video_path)

        if not Path(video_path).exists():
            logger.error("Video not found for post-processing: %s", video_path)
            return None

        # Pre-validate: corrupt MP4s (no moov atom) crash FFMPEG at the
        # native level — Python try/except cannot catch that.
        if not self._has_moov_atom(video_path):
            logger.error(
                "Video is corrupt (no moov atom), skipping: %s", video_path
            )
            return "CORRUPT"

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("Failed to open video: %s", video_path)
            return None

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                logger.error("Video has no frames: %s", video_path)
                return None

            step = max(1, total_frames // 20)

            best_frame = None
            best_conf = 0.0
            best_box = None  # Track the gold bounding box (x1, y1, x2, y2)
            fallback_frame = None

            for idx in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Save a fallback at ~30% of video
                if fallback_frame is None and idx >= int(total_frames * 0.3):
                    fallback_frame = frame.copy()

                # Blur check
                blur = self.compute_blur_score(frame)
                if blur < 100:
                    continue

                # Gold detection
                try:
                    results = self.model(frame, conf=0.2, verbose=False)[0]
                    if results.boxes is not None and len(results.boxes) > 0:
                        # Find the box with highest confidence
                        max_idx = results.boxes.conf.argmax().item()
                        max_conf = results.boxes.conf[max_idx].item()
                        if max_conf > best_conf:
                            best_conf = max_conf
                            best_frame = frame.copy()
                            # Store bounding box coords
                            best_box = results.boxes.xyxy[max_idx].cpu().numpy().astype(int)
                except Exception as e:
                    logger.error("Frame extraction inference error: %s", e)
                    continue

            # Fallback: grab frame at 30% if no good detection found
            if best_frame is None:
                if fallback_frame is not None:
                    best_frame = fallback_frame
                    logger.info("Using fallback frame at 30%% for %s", video_path)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.3))
                    ret, frame = cap.read()
                    if ret:
                        best_frame = frame
                        logger.info("Using 30%% position frame for %s", video_path)
                    else:
                        logger.error("Frame extraction failed: no readable frames in %s", video_path)
                        return None

            # Crop to gold bounding box if we have one
            if best_box is not None:
                h, w = best_frame.shape[:2]
                bx1, by1, bx2, by2 = best_box
                # Add 10% padding around the gold piece
                pad_x = int((bx2 - bx1) * 0.10)
                pad_y = int((by2 - by1) * 0.10)
                bx1 = max(0, bx1 - pad_x)
                by1 = max(0, by1 - pad_y)
                bx2 = min(w, bx2 + pad_x)
                by2 = min(h, by2 + pad_y)
                best_frame = best_frame[by1:by2, bx1:bx2]
                logger.info("Cropped to gold region: %dx%d", bx2 - bx1, by2 - by1)

            # Save
            stem = Path(video_path).stem
            out_path = self.output_dir / f"{stem}_gold.jpg"
            cv2.imwrite(str(out_path), best_frame)
            logger.info("Image extraction complete: %s (conf=%.2f)", out_path, best_conf)
            return str(out_path)

        except Exception as e:
            logger.error("Frame extraction failed: %s", e)
            return None
        finally:
            cap.release()
