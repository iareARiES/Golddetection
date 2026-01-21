"""
Simple YOLO detection script.

Simple webcam detection with Gold.pt. Runs on camera index 0, shows labels and
confidence scores, and prints detections to the console. Saving is off by
default for webcam.
"""

from pathlib import Path

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
    conf = 0.25
    imgsz = 640
    show = True
    save = False  # off for webcam

    results = model.predict(
        source=source,
        conf=conf,
        imgsz=imgsz,
        stream=True,  # webcam streaming
        show=show,
        save=save,
        device=None,  # auto device
    )

    # Class names from the model (used for readable labels)
    names = getattr(model, "names", None)

    # Collect output directories from results (list or generator)
    out_dirs = set()
    for r in results:
        # Print detections with labels and confidence, especially for gold items
        boxes = getattr(r, "boxes", None)
        if boxes is not None and getattr(boxes, "cls", None) is not None:
            cls_ids = boxes.cls.tolist()
            confs = boxes.conf.tolist()
            for cls_id, conf in zip(cls_ids, confs):
                label = names[int(cls_id)] if names is not None else str(int(cls_id))
                print(f"Detected {label} with confidence {conf:.2f}")

        if hasattr(r, "save_dir") and r.save_dir:
            out_dirs.add(Path(r.save_dir).resolve())

    if save and out_dirs:
        print("\nDetections saved to:")
        for d in sorted(out_dirs):
            print(f"  {d}")
    else:
        print("\nWebcam mode finished.")


if __name__ == "__main__":
    main()
