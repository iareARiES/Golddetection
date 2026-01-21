"""
Simple YOLO detection script.

Simple webcam detection with Gold.pt. Runs on camera index 0, shows labels and
confidence scores, and prints detections to the console. Saving is off by
default for webcam.
"""

from pathlib import Path
from datetime import datetime
import time

import cv2

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "ultralytics is required. Install with: pip install ultralytics"
    ) from exc


def main() -> None:
    weights_path = Path("Gold.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    # Fixed webcam source and simple defaults
    source = 0  # webcam index
    conf = 0.1
    imgsz = 640
    show = True
    save = False  # off for webcam

    results = model.predict(
        source=source,
        conf=conf,
        imgsz=imgsz,
        stream=True,  # webcam streaming
        show=False,  # manual display via cv2
        save=False,  # manual recording
        device=None,  # auto device
    )

    # Class names from the model (used for readable labels)
    names = getattr(model, "names", None)
    if names:
        # Force all class labels to display as "Gold"
        for k in names.keys():
            names[k] = "Gold"

    # Recording setup
    recording = False
    writer = None
    out_file = None
    out_dir = Path("runs/recordings")
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    last_detection_time = 0  # timestamp of last detection

    # Collect output directories from results (list or generator)
    out_dirs = set()
    for r in results:
        current_time = time.time()
        gold_detected = False

        # Print detections with labels and confidence, especially for gold items
        boxes = getattr(r, "boxes", None)
        gold_detected = False
        if boxes is not None and getattr(boxes, "cls", None) is not None:
            cls_ids = boxes.cls.tolist()
            confs = boxes.conf.tolist()
            for cls_id, conf in zip(cls_ids, confs):
                label = "Gold"
                print(f"Detected {label} with confidence {conf:.2f}")
                gold_detected = True
                last_detection_time = current_time

        # Get the original frame and manually draw boxes with "Gold" labels
        frame = r.orig_img.copy()
        if boxes is not None and len(boxes) > 0:
            for box, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Gold {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Start recording if gold detected and not already recording
        if gold_detected and not recording:
            h, w = frame.shape[:2]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = out_dir / f"gold_detected_{ts}.mp4"
            writer = cv2.VideoWriter(str(out_file), fourcc, 20.0, (w, h))
            recording = True
            print(f"Recording started: {out_file}")

        # Stop recording if no detection for 30 seconds
        if recording and (current_time - last_detection_time) > 30:
            if writer is not None:
                writer.release()
                writer = None
                print(f"Recording stopped: {out_file}")
            recording = False

        # Add status text overlays
        status1 = f"Gold Detected: {'Yes' if gold_detected else 'No'}"
        status2 = f"Recording: {'On' if recording else 'Off'}"
        cv2.putText(frame, status1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, status2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if recording and writer is not None:
            writer.write(frame)

        if show:
            cv2.imshow("Gold Detection", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        if hasattr(r, "save_dir") and r.save_dir:
            out_dirs.add(Path(r.save_dir).resolve())

    if writer is not None:
        writer.release()
        print(f"Recording stopped: {out_file}")
    cv2.destroyAllWindows()

    if save and out_dirs:
        print("\nDetections saved to:")
        for d in sorted(out_dirs):
            print(f"  {d}")
    else:
        print("\nWebcam mode finished.")


if __name__ == "__main__":
    main()
