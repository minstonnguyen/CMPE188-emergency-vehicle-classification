"""Live video inference: YOLO + classifier, both on OpenVINO GPU.

Inference runs on a background thread; display runs at native video FPS.
The threads share a list of (x1,y1,x2,y2,label,color) detections via a lock.

Usage (from project root):
    python app/detect_video.py app/video.mp4
    python app/detect_video.py app/video.mp4 --device CPU --conf 0.45
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

_RED   = (0, 0, 255)
_GREEN = (0, 255, 0)

_CLS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_CLS_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

YOLO_SIZE = 640
YOLO_ONNX = Path("models/yolo/yolov5m_relu6_car--640x640_float_openvino_multidevice_1.onnx")
CLS_ONNX  = Path("models/best_model.onnx")


# ---------------------------------------------------------------------------
# YOLO helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88).astype(np.float32)))


def letterbox(frame_bgr: np.ndarray, size: int = YOLO_SIZE):
    h, w = frame_bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_top  = (size - nh) // 2
    pad_left = (size - nw) // 2
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = cv2.resize(
        frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    return canvas, r, pad_left, pad_top


def yolo_preprocess(frame_bgr: np.ndarray):
    canvas, r, pl, pt = letterbox(frame_bgr)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[np.newaxis], r, pl, pt


def yolo_postprocess(
    preds: np.ndarray,
    grid_xy: np.ndarray,
    anchor_wh: np.ndarray,
    strides: np.ndarray,
    r: float, pad_left: int, pad_top: int,
    orig_h: int, orig_w: int,
    conf_thresh: float, iou_thresh: float,
) -> list[tuple[int, int, int, int, float]]:
    sig_xy = _sigmoid(preds[:, 0:2])
    sig_wh = _sigmoid(preds[:, 2:4])

    # grid_xy is already in pixel space — do NOT multiply by stride again
    cx = (sig_xy[:, 0] * 2.0 - 0.5) * strides + grid_xy[:, 0]
    cy = (sig_xy[:, 1] * 2.0 - 0.5) * strides + grid_xy[:, 1]
    w  = (sig_wh[:, 0] * 2.0) ** 2 * anchor_wh[:, 0]
    h  = (sig_wh[:, 1] * 2.0) ** 2 * anchor_wh[:, 1]

    scores = _sigmoid(preds[:, 4]) * _sigmoid(preds[:, 5])

    mask = scores >= conf_thresh
    cx, cy, w, h, scores = cx[mask], cy[mask], w[mask], h[mask], scores[mask]
    if len(scores) == 0:
        return []

    x1 = cx - w / 2
    y1 = cy - h / 2
    indices = cv2.dnn.NMSBoxes(
        np.stack([x1, y1, w, h], axis=1).tolist(),
        scores.tolist(), conf_thresh, iou_thresh,
    )
    if len(indices) == 0:
        return []

    results = []
    for i in np.array(indices).flatten():
        bx1 = int(np.clip((x1[i]        - pad_left) / r, 0, orig_w))
        by1 = int(np.clip((y1[i]        - pad_top)  / r, 0, orig_h))
        bx2 = int(np.clip((x1[i] + w[i] - pad_left) / r, 0, orig_w))
        by2 = int(np.clip((y1[i] + h[i] - pad_top)  / r, 0, orig_h))
        if bx2 > bx1 and by2 > by1:
            results.append((bx1, by1, bx2, by2, float(scores[i])))
    return results


# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------

def cls_preprocess(crop_bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - _CLS_MEAN) / _CLS_STD
    return arr.transpose(2, 0, 1)[np.newaxis]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_box(frame, x1, y1, x2, y2, label, color, thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    label_y = max(y1, th + 6)
    cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 4, label_y + baseline), color, -1)
    cv2.putText(frame, label, (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------

def load_class_names(onnx_path: Path, override: list[str] | None) -> list[str]:
    if override:
        return override
    companion = onnx_path.with_suffix(".pt")
    if companion.exists():
        import torch
        try:
            ckpt = torch.load(companion, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(companion, map_location="cpu")
        names = ckpt.get("class_names")
        if names:
            return list(names)
    return ["emergency_vehicle", "non_emergency"]


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------

def inference_worker(
    frame_q: queue.Queue,
    det_lock: threading.Lock,
    det_ref: list,
    stop_evt: threading.Event,
    yolo_req, yolo_inp_node,
    grid_xy: np.ndarray,
    anchor_wh: np.ndarray,
    strides: np.ndarray,
    cls_req, cls_inp_node,
    cls_size: int,
    class_names: list[str],
    emg_idx: int,
    conf_thresh: float,
    iou_thresh: float,
):
    while not stop_evt.is_set():
        try:
            frame = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        orig_h, orig_w = frame.shape[:2]
        inp, r, pl, pt = yolo_preprocess(frame)
        yolo_req.infer({yolo_inp_node: inp})
        preds = yolo_req.get_output_tensor(0).data[0]   # [25200, 6]

        boxes = yolo_postprocess(
            preds, grid_xy, anchor_wh, strides,
            r, pl, pt, orig_h, orig_w,
            conf_thresh, iou_thresh,
        )

        crops, valid = [], []
        for (x1, y1, x2, y2, _) in boxes:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(cls_preprocess(crop, cls_size))
            valid.append((x1, y1, x2, y2))

        new_dets = []
        if crops:
            cls_req.infer({cls_inp_node: np.concatenate(crops, axis=0)})
            for (x1, y1, x2, y2), logits in zip(valid, cls_req.get_output_tensor(0).data):
                probs = softmax(logits)
                pred  = int(np.argmax(probs))
                conf  = float(probs[pred])
                color = _RED if pred == emg_idx else _GREEN
                new_dets.append((x1, y1, x2, y2, f"{class_names[pred]}  {conf:.0%}", color))

        with det_lock:
            det_ref.clear()
            det_ref.extend(new_dets)


# ---------------------------------------------------------------------------
# Args + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video",        type=Path)
    parser.add_argument("--yolo",       type=Path,  default=YOLO_ONNX)
    parser.add_argument("--classifier", type=Path,  default=CLS_ONNX)
    parser.add_argument("--device",     default="GPU", help="OpenVINO device for both models.")
    parser.add_argument("--conf",       type=float, default=0.45)
    parser.add_argument("--iou",        type=float, default=0.45)
    parser.add_argument("--warmup",     type=int,   default=5)
    parser.add_argument("--classes",    nargs="+",  default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for p in (args.video, args.yolo, args.classifier):
        if not p.exists():
            sys.exit(f"[ERROR] Not found: {p}")

    try:
        import openvino as ov
    except ImportError:
        sys.exit("[ERROR] pip install openvino")

    core   = ov.Core()
    avail  = core.available_devices
    device = args.device
    if device not in avail and device != "AUTO":
        print(f"[WARN] '{device}' not available {avail}. Falling back to CPU.")
        device = "CPU"
    print(f"Device : {device}\n")

    # YOLO
    print(f"Loading YOLO       : {args.yolo}")
    yolo_compiled = core.compile_model(core.read_model(str(args.yolo)), device)
    yolo_inp_node = yolo_compiled.input(0)
    yolo_req      = yolo_compiled.create_infer_request()

    # Cache static metadata with one dummy pass
    print("Caching YOLO grid/anchor/stride ...")
    dummy_yolo = np.zeros((1, 3, YOLO_SIZE, YOLO_SIZE), dtype=np.float32)
    yolo_req.infer({yolo_inp_node: dummy_yolo})
    anchor_wh = yolo_req.get_output_tensor(1).data.copy()  # [25200,2] — 9 unique anchor sizes
    grid_xy   = yolo_req.get_output_tensor(2).data.copy()  # [25200,2] — pixel-space grid centres
    strides   = yolo_req.get_output_tensor(3).data.copy()  # [25200]   — per-anchor stride
    print(f"  strides : {np.unique(strides).tolist()}")

    # Classifier
    print(f"Loading classifier : {args.classifier}")
    cls_compiled = core.compile_model(core.read_model(str(args.classifier)), device)
    cls_inp_node = cls_compiled.input(0)
    cls_req      = cls_compiled.create_infer_request()
    cls_shape    = cls_compiled.input(0).partial_shape
    cls_size     = cls_shape[2].get_length() if cls_shape[2].is_static else 224

    class_names = load_class_names(args.classifier, args.classes)
    emg_idx = next(
        (i for i, n in enumerate(class_names) if "emergency" in n.lower() and "non" not in n.lower()),
        0,
    )
    print(f"Classes : {class_names}  (emergency idx={emg_idx})")

    # Warm up
    print(f"\nWarming up ({args.warmup} passes each) ...", end=" ", flush=True)
    dummy_cls = np.zeros((1, 3, cls_size, cls_size), dtype=np.float32)
    for _ in range(args.warmup):
        yolo_req.infer({yolo_inp_node: dummy_yolo})
        cls_req.infer({cls_inp_node: dummy_cls})
    print("done\n")

    # Shared state
    frame_q  = queue.Queue(maxsize=1)
    det_lock = threading.Lock()
    det_ref  = []
    stop_evt = threading.Event()

    worker = threading.Thread(
        target=inference_worker,
        args=(
            frame_q, det_lock, det_ref, stop_evt,
            yolo_req, yolo_inp_node,
            grid_xy, anchor_wh, strides,
            cls_req, cls_inp_node, cls_size,
            class_names, emg_idx,
            args.conf, args.iou,
        ),
        daemon=True,
    )
    worker.start()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: {args.video}")

    video_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / video_fps

    window = "Emergency Vehicle Detector  [q=quit  space=pause]"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    disp_times: list[float] = []
    paused = False

    while True:
        t0 = time.perf_counter()

        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                frame_q.put_nowait(frame.copy())
            except queue.Full:
                pass

            with det_lock:
                current = list(det_ref)

            for (x1, y1, x2, y2, label, color) in current:
                draw_box(frame, x1, y1, x2, y2, label, color)

            disp_times.append(time.perf_counter() - t0)
            if len(disp_times) > 60:
                disp_times.pop(0)
            fps = 1.0 / (sum(disp_times) / len(disp_times))
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Cars: {len(current)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, frame)

        # Throttle display to native video FPS
        elapsed = time.perf_counter() - t0
        wait_ms = max(1, int((frame_delay - elapsed) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

    stop_evt.set()
    worker.join(timeout=3)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
