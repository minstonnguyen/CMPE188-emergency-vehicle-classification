"""Diagnostic script to understand the YOLO model output format.

Runs a single forward pass with a real image (or a test pattern if no image
given) and prints everything needed to figure out the correct decode formula.

Usage:
    python scripts/test_yolo_output.py
    python scripts/test_yolo_output.py path/to/image.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

YOLO_ONNX = Path("models/yolo/yolov5m_relu6_car--640x640_float_openvino_multidevice_1.onnx")
YOLO_SIZE  = 640


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float32), -88, 88)))


def letterbox(img_bgr, size=YOLO_SIZE):
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_top  = (size - nh) // 2
    pad_left = (size - nw) // 2
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = cv2.resize(img_bgr, (nw, nh))
    return canvas, r, pad_left, pad_top


def sep(title=""):
    w = 70
    if title:
        print(f"\n{'─'*3} {title} {'─'*(w - len(title) - 5)}")
    else:
        print("─" * w)


def main():
    import openvino as ov

    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    # Load model
    sep("Model")
    core     = ov.Core()
    model    = core.read_model(str(YOLO_ONNX))
    compiled = core.compile_model(model, "CPU")   # CPU for deterministic debug
    req      = compiled.create_infer_request()

    print(f"Model   : {YOLO_ONNX.name}")
    for i, inp in enumerate(compiled.inputs):
        print(f"Input [{i}] : {inp.partial_shape}  {inp.element_type}")
    for i, out in enumerate(compiled.outputs):
        print(f"Output[{i}] : {out.partial_shape}  {out.element_type}")

    # Prepare input
    sep("Input image")
    if image_path and image_path.exists():
        frame = cv2.imread(str(image_path))
        print(f"Loaded  : {image_path}  shape={frame.shape}")
    else:
        # Synthetic: grey frame with a white rectangle (car-like)
        frame = np.full((360, 640, 3), 100, dtype=np.uint8)
        cv2.rectangle(frame, (150, 100), (490, 280), (220, 220, 220), -1)
        print("Using synthetic test frame (grey background, white rectangle)")

    canvas, r, pad_left, pad_top = letterbox(frame)
    rgb  = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp  = rgb.transpose(2, 0, 1)[np.newaxis]
    print(f"Tensor  : {inp.shape}  [{inp.min():.3f}, {inp.max():.3f}]")

    # Infer
    req.infer({compiled.input(0): inp})

    raw      = req.get_output_tensor(0).data   # [1, 25200, 6]
    anch_wh  = req.get_output_tensor(1).data   # [25200, 2]  — anchor WH (9 unique sizes)
    grid_xy  = req.get_output_tensor(2).data   # [25200, 2]  — pixel-space grid centres
    strides  = req.get_output_tensor(3).data   # [25200]     — per-anchor stride
    preds    = raw[0]                          # [25200, 6]

    # ── Raw output stats ────────────────────────────────────────────────────
    sep("Raw output[0]  [25200, 6]  (before sigmoid)")
    for c, name in enumerate(["tx", "ty", "tw", "th", "obj_logit", "cls_logit"]):
        col = preds[:, c]
        print(f"  col[{c}] {name:<12}: min={col.min():8.3f}  max={col.max():8.3f}  "
              f"mean={col.mean():8.3f}  std={col.std():.3f}")

    sep("After sigmoid")
    sig = _sigmoid(preds)
    for c, name in enumerate(["sx", "sy", "sw", "sh", "obj_conf", "cls_conf"]):
        col = sig[:, c]
        print(f"  col[{c}] {name:<12}: min={col.min():.4f}  max={col.max():.4f}  "
              f"mean={col.mean():.4f}")

    # ── Metadata ─────────────────────────────────────────────────────────────
    sep("output[1]  grid_xy  [25200, 2]")
    print(f"  x : min={grid_xy[:,0].min():.1f}  max={grid_xy[:,0].max():.1f}  "
          f"unique count={len(np.unique(grid_xy[:,0]))}")
    print(f"  y : min={grid_xy[:,1].min():.1f}  max={grid_xy[:,1].max():.1f}  "
          f"unique count={len(np.unique(grid_xy[:,1]))}")

    sep("output[2]  anchor_wh  [25200, 2]")
    print(f"  w : min={anch_wh[:,0].min():.1f}  max={anch_wh[:,0].max():.1f}  "
          f"unique vals={sorted(np.unique(anch_wh[:,0]).tolist())}")
    print(f"  h : min={anch_wh[:,1].min():.1f}  max={anch_wh[:,1].max():.1f}  "
          f"unique vals={sorted(np.unique(anch_wh[:,1]).tolist())}")

    sep("output[3]  strides  [25200]")
    print(f"  unique={np.unique(strides).tolist()}")
    for s in np.unique(strides):
        n = (strides == s).sum()
        print(f"    stride={int(s):2d} → {n} anchors  "
              f"(grid {int(n/3)}={int(np.sqrt(n/3))}×{int(np.sqrt(n/3))}×3)")

    # ── Confidence distribution ───────────────────────────────────────────────
    sep("Confidence distribution  (obj_conf × cls_conf)")
    scores = _sigmoid(preds[:, 4]) * _sigmoid(preds[:, 5])
    for t in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        n = (scores >= t).sum()
        print(f"  >= {t:.2f} : {n:5d} anchors")

    def _fmt(arr, m):
        return f"{arr[m].min():.1f} – {arr[m].max():.1f}" if m.any() else "n/a (no detections)"

    sig_xy = _sigmoid(preds[:, 0:2])
    sig_wh = _sigmoid(preds[:, 2:4])
    w_dec  = (sig_wh[:, 0] * 2) ** 2 * anch_wh[:, 0]
    h_dec  = (sig_wh[:, 1] * 2) ** 2 * anch_wh[:, 1]

    # Use a lower threshold so we see something even on easy frames
    for thresh in (0.3, 0.1, 0.01):
        mask = scores >= thresh
        if mask.any():
            break
    print(f"  (using conf>={thresh} — {mask.sum()} anchors)")

    # ── Formula A: (sig*2 - 0.5 + grid_idx) * stride  (grid=cell index) ─────
    sep("Formula A  — grid_xy as cell index, multiply by stride")
    cx_A = (sig_xy[:, 0] * 2 - 0.5 + grid_xy[:, 0]) * strides
    cy_A = (sig_xy[:, 1] * 2 - 0.5 + grid_xy[:, 1]) * strides
    print(f"  cx : {_fmt(cx_A, mask)}  (expect 0–640)")
    print(f"  cy : {_fmt(cy_A, mask)}  (expect 0–640)")
    print(f"  w  : {_fmt(w_dec, mask)}")
    print(f"  h  : {_fmt(h_dec, mask)}")

    # ── Formula B: (sig*2 - 0.5)*stride + grid_px  (grid=pixel space) ────────
    sep("Formula B  — grid_xy already in pixel space")
    cx_B = (sig_xy[:, 0] * 2 - 0.5) * strides + grid_xy[:, 0]
    cy_B = (sig_xy[:, 1] * 2 - 0.5) * strides + grid_xy[:, 1]
    print(f"  cx : {_fmt(cx_B, mask)}  (expect 0–640)")
    print(f"  cy : {_fmt(cy_B, mask)}  (expect 0–640)")
    print(f"  w  : {_fmt(w_dec, mask)}")
    print(f"  h  : {_fmt(h_dec, mask)}")

    # ── Check grid_xy divisibility by stride ─────────────────────────────────
    sep("Is grid_xy divisible by its stride? (tells us if it's pixel-space)")
    for s_val in np.unique(strides):
        idx = strides == s_val
        gx  = grid_xy[idx, 0]
        gy  = grid_xy[idx, 1]
        div_x = np.all(gx % s_val == 0)
        div_y = np.all(gy % s_val == 0)
        print(f"  stride={int(s_val):2d}: grid_x divisible={div_x}  grid_y divisible={div_y}  "
              f"(if True → pixel-space, already × stride)")
        # Show a few sample grid values for this stride
        samples = np.unique(gx)[:8]
        print(f"            sample grid_x values: {samples.tolist()}")

    # ── Show top-5 detections ─────────────────────────────────────────────────
    sep(f"Top-5 detections (conf>={thresh}) — Formula B")
    top_idx = np.argsort(scores)[::-1]
    count = 0
    orig_h, orig_w = frame.shape[:2]
    for i in top_idx:
        if scores[i] < 0.3:
            break
        x1 = (cx_B[i] - w_B[i]/2 - pad_left) / r
        y1 = (cy_B[i] - h_B[i]/2 - pad_top)  / r
        x2 = (cx_B[i] + w_B[i]/2 - pad_left) / r
        y2 = (cy_B[i] + h_B[i]/2 - pad_top)  / r
        in_frame = (0 <= x1 < orig_w and 0 <= y1 < orig_h and
                    0 <= x2 <= orig_w and 0 <= y2 <= orig_h and x2>x1 and y2>y1)
        print(f"  [{i:5d}] score={scores[i]:.4f}  stride={int(strides[i]):2d}  "
              f"frame=({x1:.0f},{y1:.0f})–({x2:.0f},{y2:.0f})  "
              f"size={x2-x1:.0f}×{y2-y1:.0f}  in_frame={in_frame}")
        count += 1
        if count >= 5:
            break

    sep()
    print("Done. Use the output above to determine the correct decode formula.")
    print("The correct formula produces cx/cy in [0, 640] for conf>=0.3 anchors.")


if __name__ == "__main__":
    main()
