"""
Gold Detection Pipeline v4.1 — Natural Screen, Mask-Only Shine Tracking
------------------------------------------------------------------------
Pipeline:
  Stage 1: YOLO confidence gate    (conf > 0.25)
  Stage 2: HSV yellow gate         (is it yellow-gold at all?)
  Stage 3: Luminosity sweep        (shine tracked INSIDE mask only)
            → ΔL* HIGH + Δb* HIGH  = GOLD
            → ΔL* HIGH + Δb* LOW   = PLASTIC
            → ΔL* LOW              = FABRIC

Camera: fully natural auto-exposure — no locking, no darkening.
Shine detection reads brightness only from the segmented mask pixels,
so camera auto-exposure adjusting the whole frame does not interfere.

Controls:
  Q = quit  S = save  D = debug  R = reset
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto
import threading
import time

# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Gold Detection v4.1")
parser.add_argument("--model",            default=r"E:\Staunch\GoldSegmentationTraining\best.pt")
parser.add_argument("--conf",             type=float, default=0.25)
parser.add_argument("--cam",              type=int,   default=0)
# HSV yellow gate — very wide, catches all gold shades
parser.add_argument("--hsv-h-lo",         type=int,   default=0,
                    help="Hue lower (0 = catches rose gold wrapping)")
parser.add_argument("--hsv-h-hi",         type=int,   default=50,
                    help="Hue upper (50 = catches orange-gold)")
parser.add_argument("--hsv-s-min",        type=int,   default=35,
                    help="Min saturation (35 = dim/brushed gold)")
parser.add_argument("--hsv-v-min",        type=int,   default=35,
                    help="Min value (35 = shadowed/antique gold)")
parser.add_argument("--hsv-coverage",     type=float, default=0.10,
                    help="Min fraction yellow (10% = partial gold)")
# TSD thresholds — relaxed for indoor / matte gold
parser.add_argument("--min-dl",           type=float, default=7.0,
                    help="Min ΔL* at peak (7 = catches weak shine)")
parser.add_argument("--min-db",           type=float, default=2.5,
                    help="Min Δb* at peak (2.5 = catches subtle gold tint)")
# Sweep detection — sensitive to mild torch movement
parser.add_argument("--sweep-rise",       type=float, default=4.0,
                    help="ΔL* rise to detect flash (4 = mild torch)")
parser.add_argument("--sweep-fall",       type=float, default=2.5,
                    help="ΔL* drop from peak to confirm sweep passed")
parser.add_argument("--sweep-min-frames", type=int,   default=6,
                    help="Min frames above baseline for valid sweep")
parser.add_argument("--sweep-timeout",    type=int,   default=300,
                    help="Frames before timeout (~10s at 30fps)")
args = parser.parse_args()

GOLD_CLASS_ID = 0


# ─────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────
class ObjState(Enum):
    PENDING  = auto()   # yellow gate passed, thread not started yet
    REJECTED = auto()   # failed yellow gate
    SWEEPING = auto()   # thread watching, waiting for flash sweep
    GOLD     = auto()   # confirmed gold
    PLASTIC  = auto()   # shine went white — not gold
    FABRIC   = auto()   # no shine — matte/fabric
    TIMEOUT  = auto()   # no sweep detected in time


LABEL = {
    ObjState.PENDING:  "PENDING",
    ObjState.REJECTED: "NOT YELLOW",
    ObjState.SWEEPING: "SWEEPING — shine flash now",
    ObjState.GOLD:     "GOLD",
    ObjState.PLASTIC:  "PLASTIC (white shine)",
    ObjState.FABRIC:   "FABRIC (no shine)",
    ObjState.TIMEOUT:  "TIMEOUT",
}

COL = {
    ObjState.PENDING:  (120, 120, 120),
    ObjState.REJECTED: ( 50,  50,  50),
    ObjState.SWEEPING: (  0, 200,  80),
    ObjState.GOLD:     (  0, 215, 255),
    ObjState.PLASTIC:  ( 30,  30, 200),
    ObjState.FABRIC:   (160,  60, 160),
    ObjState.TIMEOUT:  ( 80,  80,  80),
}


@dataclass
class SweepSample:
    mask_L:  float
    mask_b:  float   # mean b* of entire mask (gold colour retention)
    spec_L:  float
    spec_a:  float
    spec_b:  float


@dataclass
class TSDResult:
    state:     ObjState = ObjState.PENDING
    peak_dl:   float    = 0.0
    peak_db:   float    = 0.0
    tsd_score: float    = 0.0
    reason:    str      = ""
    n_samples: int      = 0


@dataclass
class DetectionRecord:
    track_id:     int
    box:          tuple
    mask:         np.ndarray
    yolo_conf:    float
    hsv_pass:     bool
    hsv_coverage: float
    state:        ObjState      = ObjState.PENDING
    tsd:          TSDResult     = field(default_factory=TSDResult)
    sweep_thread: Optional[threading.Thread] = field(default=None, repr=False)
    thread_lock:  threading.Lock = field(default_factory=threading.Lock)
    last_seen:    int            = 8


# ─────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────
detections:           list[DetectionRecord] = []
det_lock              = threading.Lock()
next_track_id         = 1
debug_overlay         = True
screenshot_count      = 0
latest_lab_frame      = None
latest_lab_lock       = threading.Lock()


# ─────────────────────────────────────────────────────────────
# Mask-only brightness utilities
# (whole frame is NEVER used — only pixels inside segmented mask)
# ─────────────────────────────────────────────────────────────
def mask_mean_L(lab, mask):
    """Mean L* of pixels inside mask only."""
    px = lab[mask].astype(np.float32)
    return float(np.mean(px[:, 0])) if len(px) > 0 else 0.0


def mask_mean_Lab(lab, mask):
    """Mean L*, a*, b* of ALL pixels inside mask.
    Unlike specular stats, this doesn't blow out — gives stable
    colour measurements even when highlights are saturated."""
    px = lab[mask].astype(np.float32)
    if len(px) == 0:
        return 0.0, 128.0, 128.0
    return (float(np.mean(px[:, 0])),
            float(np.mean(px[:, 1])),
            float(np.mean(px[:, 2])))


def specular_stats(lab, mask, top_pct=40.0):
    """
    Top top_pct% brightest pixels inside mask → mean L*, a*, b*.
    Excludes blown-out pixels (L > 252) to avoid saturation artifacts.
    Returns None if not enough data.
    """
    px = lab[mask].astype(np.float32)
    if len(px) < 20:
        return None
    l  = px[:, 0]
    th = np.percentile(l, 100.0 - top_pct)
    hi = px[(l >= th) & (l < 252)]
    if len(hi) < 5:
        return None
    return float(np.mean(hi[:, 0])), float(np.mean(hi[:, 1])), float(np.mean(hi[:, 2]))


# ─────────────────────────────────────────────────────────────
# Stage 1 — HSV yellow gate
# ─────────────────────────────────────────────────────────────
def check_hsv_yellow(hsv, mask):
    """
    Is the object yellow-gold coloured?

    Very wide gate — catches all gold shades:
      Rose gold / pink    H: 0–18
      Standard yellow     H: 18–30
      Orange-yellow       H: 30–50
      Antique / oxidized  covered by warm fallback

    Returns (passed, coverage_fraction)
    """
    px = hsv[mask]
    if len(px) < 50:
        return False, 0.0

    H = px[:, 0].astype(np.float32)
    S = px[:, 1].astype(np.float32)
    V = px[:, 2].astype(np.float32)

    gold = (H >= args.hsv_h_lo) & (H <= args.hsv_h_hi)
    rose = H <= 5   # rose gold wraps near hue 0

    yellow = (
        (gold | rose) &
        (S >= args.hsv_s_min) &
        (V >= args.hsv_v_min)
    )

    # Extra warm-yellow tolerance — catches antique, brushed, oxidized gold
    warm = (
        (H >= 0) & (H <= 60) &
        (S >= 25) &
        (V >= 30)
    )
    yellow = yellow | warm

    cov = float(np.mean(yellow))
    return cov >= args.hsv_coverage, cov


# ─────────────────────────────────────────────────────────────
# Stage 2 — Sweep worker thread (one per detected object)
# ─────────────────────────────────────────────────────────────
def sweep_worker(det: DetectionRecord):
    """
    Monitors shine response inside the object mask continuously.

    Phase 1 — Baseline (15 frames):
      Quietly measures ambient mask brightness before any flash.
      Sets baseline_L and baseline_spec_b.

    Phase 2 — Wait for flash (rising edge):
      Every frame: is mask_L > baseline + sweep_rise?
      No  → keep waiting (up to sweep_timeout)
      Yes → flash arrived, record samples

    Phase 3 — Record sweep:
      Track brightest moment (peak specular sample).

    Phase 4 — Falling edge:
      mask_L drops > sweep_fall from peak AND min samples collected
      → sweep done → decision from peak

    Decision:
      peak_ΔL* < min_dl          → FABRIC  (absorbed light, no shine)
      peak_ΔL* ≥ min_dl,
        peak_Δb* < min_db        → PLASTIC (shine went white)
      peak_ΔL* ≥ min_dl,
        peak_Δb* ≥ min_db        → GOLD    (shine stayed gold-coloured)
    """
    BASELINE_FRAMES = 15

    baseline_samples = []
    sweep_samples    = []
    baseline_L       = None
    baseline_mask_b  = None   # mask-wide mean b* at baseline
    in_sweep         = False
    peak_sample      = None
    frame_count      = 0

    print(f"[T#{det.track_id}] Sweep thread started")

    while True:
        with latest_lab_lock:
            lab = latest_lab_frame
        if lab is None:
            time.sleep(0.01)
            continue

        # Read brightness INSIDE mask only — camera auto-exposure irrelevant
        spec = specular_stats(lab, det.mask)
        ml   = mask_mean_L(lab, det.mask)
        _, _, mb = mask_mean_Lab(lab, det.mask)   # mask-wide mean b*
        frame_count += 1

        # Phase 1 — baseline
        if baseline_L is None:
            baseline_samples.append((ml, mb))
            if len(baseline_samples) >= BASELINE_FRAMES:
                baseline_L      = float(np.mean([m for m, _ in baseline_samples]))
                baseline_mask_b = float(np.mean([b for _, b in baseline_samples]))
                print(f"[T#{det.track_id}] Baseline: maskL={baseline_L:.1f} "
                      f"maskB={baseline_mask_b:.1f}")
            time.sleep(0.033)
            continue

        delta = ml - baseline_L

        if not in_sweep:
            if delta >= args.sweep_rise:
                in_sweep = True
                print(f"[T#{det.track_id}] Flash detected — ΔL={delta:.1f}")
            elif frame_count > args.sweep_timeout:
                with det.thread_lock:
                    det.tsd   = TSDResult(state=ObjState.TIMEOUT,
                                          reason=f"no_sweep/{frame_count}fr")
                    det.state = ObjState.TIMEOUT
                print(f"[T#{det.track_id}] Timeout")
                return
        else:
            if spec is not None:
                s = SweepSample(mask_L=ml, mask_b=mb, spec_L=spec[0],
                                spec_a=spec[1], spec_b=spec[2])
                sweep_samples.append(s)
                # Track peak by MASK brightness — specular saturates too fast
                if peak_sample is None or ml > peak_sample.mask_L:
                    peak_sample = s

            if peak_sample is not None:
                drop = peak_sample.mask_L - ml
                if (drop >= args.sweep_fall and
                        len(sweep_samples) >= args.sweep_min_frames):
                    print(f"[T#{det.track_id}] Sweep complete — "
                          f"peak specL={peak_sample.spec_L:.1f} "
                          f"specB={peak_sample.spec_b:.1f} "
                          f"n={len(sweep_samples)}")
                    break

        time.sleep(0.016)

    # Decision
    if peak_sample is None:
        with det.thread_lock:
            det.tsd   = TSDResult(state=ObjState.TIMEOUT, reason="no_peak")
            det.state = ObjState.TIMEOUT
        return

    # Use MASK-WIDE metrics, not specular — specular blows out at L*~250
    # and loses all colour info (b* → 128 regardless of material)
    peak_dl = peak_sample.mask_L - baseline_L      # mask-wide ΔL (headroom: ~120-200 baseline)
    peak_db = peak_sample.mask_b - baseline_mask_b  # mask-wide Δb (stable gold colour signal)
    raw     = (max(peak_dl, 0) * max(peak_db, 0)) / (abs(peak_dl) + 1e-6)
    score   = float(np.clip(raw / 200.0, 0.0, 1.0))

    if peak_dl < args.min_dl:
        state  = ObjState.FABRIC
        reason = f"no_shine ΔL={peak_dl:.1f}"
    elif peak_db < args.min_db:
        state  = ObjState.PLASTIC
        reason = f"white_shine Δb={peak_db:.1f}"
    else:
        state  = ObjState.GOLD
        reason = f"gold_shine ΔL={peak_dl:.1f} Δb={peak_db:.1f}"

    with det.thread_lock:
        det.tsd = TSDResult(state=state, peak_dl=peak_dl, peak_db=peak_db,
                            tsd_score=score, reason=reason,
                            n_samples=len(sweep_samples))
        det.state = state

    print(f"[T#{det.track_id}] → {state.name}  {reason}  score={score:.2f}")


def spawn_thread(det: DetectionRecord):
    t = threading.Thread(target=sweep_worker, args=(det,),
                         daemon=True, name=f"sweep-{det.track_id}")
    det.sweep_thread = t
    det.state        = ObjState.SWEEPING
    t.start()


# ─────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────
def draw_detection(img, det: DetectionRecord):
    x1, y1, x2, y2 = det.box
    col             = COL[det.state]

    # Mask colour overlay — only for meaningful states
    if det.state in (ObjState.SWEEPING, ObjState.GOLD,
                     ObjState.PLASTIC, ObjState.FABRIC):
        ov = np.zeros_like(img)
        ov[det.mask] = col
        alpha = 0.35 if det.state == ObjState.GOLD else 0.18
        cv2.addWeighted(img, 1.0, ov, alpha, 0, img)

    # Box
    cv2.rectangle(img, (x1, y1), (x2, y2), col,
                  2 if det.state == ObjState.GOLD else 1)

    # Label
    cv2.putText(img, f"#{det.track_id} {LABEL[det.state]}",
                (x1, max(y1 - 16, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

    # Debug line
    if debug_overlay:
        if det.state == ObjState.SWEEPING:
            sub = f"conf:{det.yolo_conf:.2f}  hsv:{det.hsv_coverage:.0%}"
        elif det.state in (ObjState.GOLD, ObjState.PLASTIC, ObjState.FABRIC):
            sub = (f"ΔL:{det.tsd.peak_dl:.1f}  Δb:{det.tsd.peak_db:.1f}  "
                   f"score:{det.tsd.tsd_score:.2f}  n:{det.tsd.n_samples}")
        elif det.state == ObjState.REJECTED:
            sub = f"yellow_cov:{det.hsv_coverage:.0%} < {args.hsv_coverage:.0%}"
        else:
            sub = det.tsd.reason
        cv2.putText(img, sub,
                    (x1, max(y1 - 4, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)

    # Gold confidence bar
    if det.state == ObjState.GOLD:
        bw     = x2 - x1
        filled = int(bw * det.tsd.tsd_score)
        cv2.rectangle(img, (x1, y2 + 2), (x2,          y2 + 6), (30, 30, 30), -1)
        cv2.rectangle(img, (x1, y2 + 2), (x1 + filled, y2 + 6), COL[ObjState.GOLD], -1)

    # Pulsing border when sweeping
    if det.state == ObjState.SWEEPING and int(time.time() * 4) % 2:
        cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), col, 2)


def draw_hud(img, dets):
    gold     = sum(1 for d in dets if d.state == ObjState.GOLD)
    sweeping = sum(1 for d in dets if d.state == ObjState.SWEEPING)

    cv2.putText(img, f"Gold: {gold}   Sweeping: {sweeping}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)

    if sweeping > 0:
        cv2.putText(img, ">>> Shine & sweep torch over the object <<<",
                    (10, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 80), 1)

    cv2.putText(img, "Q=quit  S=save  D=debug  R=reset",
                (10, 454), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)

    cv2.putText(img, "DBG:ON" if debug_overlay else "DBG:OFF",
                (590, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 100), 1)


# ─────────────────────────────────────────────────────────────
# Load model + camera
# ─────────────────────────────────────────────────────────────
print(f"[INFO] Loading model: {args.model}")
model = YOLO(args.model)
print(f"[INFO] Classes: {model.names}")

# ─────────────────────────────────────────────────────────────
# Camera setup — fully automatic, natural brightness
# ─────────────────────────────────────────────────────────────
# WHY auto-exposure is fine here:
# All shine/luminance analysis reads ONLY pixels inside the
# YOLO-segmented mask — never the whole frame. So even if the
# camera auto-adjusts brightness when a torch enters the scene,
# it does NOT affect the mask-local ΔL* and Δb* measurements.
# Manual exposure was making the screen dark for no benefit.
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # full auto-exposure
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
print("[CAM] Auto-exposure ON — natural screen brightness")
print("[CAM] Shine analysis is mask-only — auto-exposure does not interfere")

print(f"\n[INFO] Pipeline: YOLO(>{args.conf}) → HSV yellow → Mask-only shine sweep")
print(f"[INFO] HSV range H:[{args.hsv_h_lo},{args.hsv_h_hi}] S≥{args.hsv_s_min} V≥{args.hsv_v_min} cov≥{args.hsv_coverage:.0%}")
print(f"[INFO] TSD: ΔL≥{args.min_dl}  Δb≥{args.min_db}")
print("[INFO] Controls: Q=quit  S=save  D=debug  R=reset\n")

# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame     = cv2.resize(frame, (640, 480))
    lab       = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv       = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    annotated = frame.copy()   # draw on copy — original untouched

    # Push latest LAB to sweep threads (mask-only reads, not whole frame stats)
    with latest_lab_lock:
        latest_lab_frame = lab.copy()

    # YOLO
    results = model(frame, conf=args.conf,
                    classes=[GOLD_CLASS_ID], verbose=False)

    with det_lock:
        # Age detections
        for d in detections:
            d.last_seen -= 1

        if results[0].masks is not None:
            for i, raw_mask in enumerate(results[0].masks.data.cpu().numpy()):
                mask_u8 = cv2.resize(
                    (raw_mask * 255).astype(np.uint8), (640, 480),
                    interpolation=cv2.INTER_NEAREST)
                bm = mask_u8 > 127
                if np.sum(bm) < 100:
                    continue

                x1, y1, x2, y2 = map(int, results[0].boxes[i].xyxy[0])
                conf_val        = float(results[0].boxes[i].conf[0])
                box             = (x1, y1, x2, y2)

                # Match existing detection
                matched = next(
                    (d for d in detections
                     if (max(0, min(x2, d.box[2]) - max(x1, d.box[0])) *
                         max(0, min(y2, d.box[3]) - max(y1, d.box[1]))) >
                        0.35 * (x2 - x1) * (y2 - y1)),
                    None
                )

                if matched:
                    matched.box       = box
                    matched.mask      = bm
                    matched.last_seen = 8
                    matched.yolo_conf = conf_val
                else:
                    # New object
                    hsv_pass, hsv_cov = check_hsv_yellow(hsv, bm)
                    nd = DetectionRecord(
                        track_id=next_track_id,
                        box=box, mask=bm,
                        yolo_conf=conf_val,
                        hsv_pass=hsv_pass,
                        hsv_coverage=hsv_cov,
                        state=ObjState.PENDING if hsv_pass else ObjState.REJECTED,
                    )
                    next_track_id += 1
                    detections.append(nd)

                    if hsv_pass:
                        spawn_thread(nd)
                        print(f"[NEW] #{nd.track_id} yellow gate passed "
                              f"({hsv_cov:.0%}) — sweep thread started")
                    else:
                        print(f"[NEW] #{nd.track_id} failed yellow gate "
                              f"({hsv_cov:.0%})")

        # Remove aged-out (keep gold permanently)
        detections[:] = [d for d in detections
                         if d.last_seen > 0 or d.state == ObjState.GOLD]

    # Draw
    with det_lock:
        snap = list(detections)

    for det in snap:
        with det.thread_lock:
            draw_detection(annotated, det)

    draw_hud(annotated, snap)
    cv2.imshow("Gold Detection v4.1", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        fname = f"screenshot_{screenshot_count:03d}.jpg"
        cv2.imwrite(fname, annotated)
        print(f"[INFO] Saved {fname}")
        screenshot_count += 1
    elif key == ord("d"):
        debug_overlay = not debug_overlay
    elif key == ord("r"):
        with det_lock:
            detections.clear()
        next_track_id = 1
        print("[INFO] Reset")

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")
