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

        cap = cv2.VideoCapture(video_path)
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
                        max_conf = results.boxes.conf.max().item()
                        if max_conf > best_conf:
                            best_conf = max_conf
                            best_frame = frame.copy()
                except Exception as e:
                    logger.error("Frame extraction inference error: %s", e)
                    continue

            # Fallback: grab frame at 30% if no good detection found
            if best_frame is None:
                if fallback_frame is not None:
                    best_frame = fallback_frame
                    logger.info("Using fallback frame at 30%% for %s", video_path)
                else:
                    # Last resort: read the very first frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.3))
                    ret, frame = cap.read()
                    if ret:
                        best_frame = frame
                        logger.info("Using 30%% position frame for %s", video_path)
                    else:
                        logger.error("Frame extraction failed: no readable frames in %s", video_path)
                        return None

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
