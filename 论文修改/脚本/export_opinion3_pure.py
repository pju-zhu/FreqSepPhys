#!/usr/bin/env python3
"""Export compact evidence for reviewer comment 3 from one PURE record.

This exporter intentionally stops at the first frequency decomposition of the
slow and fast streams (before Mamba).  It therefore does not require the
CUDA-only Mamba extension and produces the exact fixed Gaussian decomposition
that is relevant to the reviewer question.

The output directory contains, for every 128-frame clip:
  clip_XXXX.npz  slow_input/smooth/residual, fast_input/smooth/residual,
                 slow_fs and fast_fs
  clip_XXXX_bvp.csv
  clip_XXXX_motion.csv
  manifest.csv

Run from the repository root.  The checkpoint must be the Epoch4 checkpoint
used for the paper.  This script does not run the complete trained network.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def install_cpu_mamba_stub():
    """Allow importing FreqSepPhys without CUDA Mamba extensions.

    The exporter never calls a Mamba block, because it stops before the first
    FreqSepBlock.  The stub is consequently only an import-time placeholder.
    """

    stub = types.ModuleType("mamba_ssm")

    class IdentityMamba(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, *args, **kwargs):
            return x

    stub.Mamba = IdentityMamba
    sys.modules["mamba_ssm"] = stub


def load_json(record: Path):
    files = sorted(record.rglob("*.json"), key=natural_key)
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one JSON in {record}, found {len(files)}")
    with files[0].open("r", encoding="utf-8") as f:
        data = json.load(f)
    image_meta = data.get("/Image", [])
    full_meta = data.get("/FullPackage", [])
    return files[0], image_meta, full_meta


def read_frames(record: Path):
    files = sorted(record.rglob("*.png"), key=natural_key)
    if not files:
        raise RuntimeError(f"No PNG frames found under {record}")
    frames = []
    for p in files:
        with Image.open(p) as im:
            frames.append(np.asarray(im.convert("RGB")))
    return files, np.asarray(frames)


def timestamp_array(meta, expected, fallback_fps):
    vals = [x.get("Timestamp") for x in meta if isinstance(x, dict)]
    vals = [int(x) for x in vals if x is not None]
    if len(vals) == expected:
        return np.asarray(vals, dtype=np.int64)
    if vals:
        # A mismatched metadata length is kept usable by linear interpolation.
        return np.rint(np.linspace(vals[0], vals[-1], expected)).astype(np.int64)
    step = int(round(1e9 / fallback_fps))
    return np.arange(expected, dtype=np.int64) * step


def detect_box(gray, previous, cascade):
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(boxes):
        return max(boxes, key=lambda b: int(b[2]) * int(b[3])).astype(int)
    return previous


def crop_with_box(frame, box, size=128):
    x, y, w, h = [int(v) for v in box]
    side = max(w, h)
    cx, cy = x + w / 2.0, y + h / 2.0
    xx = int(round(cx - 0.75 * side))
    yy = int(round(cy - 0.75 * side))
    ww = int(round(1.5 * side))
    hh = ww
    h0, w0 = frame.shape[:2]
    x0, y0 = max(0, xx), max(0, yy)
    x1, y1 = min(w0, xx + ww), min(h0, yy + hh)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Face crop is outside the frame")
    crop = frame[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def diff_normalized(frames):
    frames = frames.astype(np.float32)
    d = (frames[1:] - frames[:-1]) / (frames[1:] + frames[:-1] + 1e-7)
    scale = float(np.std(d))
    if scale > 0:
        d = d / scale
    return np.concatenate([d, np.zeros((1,) + d.shape[1:], dtype=np.float32)], axis=0)


def to_tchw(x):
    # [1, C, T, H, W] -> [T, C, H, W]
    return x.detach().cpu().numpy()[0].transpose(1, 0, 2, 3).astype(np.float32)


def get_decomposition(model, frames):
    d = diff_normalized(frames)
    tensor = torch.from_numpy(d.transpose(3, 0, 1, 2)[None])
    with torch.no_grad():
        x = model.ConvBlock1(tensor)
        x = model.MaxpoolSpa(x)
        x = model.ConvBlock2(x)
        x = model.ConvBlock3(x)
        x = model.MaxpoolSpa(x)
        slow = model.ConvBlock4(x)
        fast = model.ConvBlock5(x)
        slow_smooth, slow_residual = model.Block1.freq_decomposer(slow)
        fast_smooth, fast_residual = model.Block4.freq_decomposer(fast)
    return {
        "slow_input": to_tchw(slow),
        "slow_smooth": to_tchw(slow_smooth),
        "slow_residual": to_tchw(slow_residual),
        "fast_input": to_tchw(fast),
        "fast_smooth": to_tchw(fast_smooth),
        "fast_residual": to_tchw(fast_residual),
    }


def write_bvp(path, full_meta, start_ns, end_ns):
    rows = []
    for item in full_meta:
        if not isinstance(item, dict):
            continue
        ts = item.get("Timestamp")
        val = item.get("Value", {})
        if ts is None or not isinstance(val, dict):
            continue
        if start_ns <= int(ts) <= end_ns:
            rows.append((int(ts), val.get("waveform", ""), val.get("pulseRate", "")))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "waveform", "pulse_rate_bpm"])
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=Path, required=True, help="PURE record directory containing PNG and JSON")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Epoch4 FreqSepPhys checkpoint")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dataset", default="PURE")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--record-id", required=True)
    ap.add_argument("--condition", default="unknown")
    ap.add_argument("--clip-length", type=int, default=128)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--max-clips", type=int, default=0, help="0 means all complete clips")
    ap.add_argument("--fallback-fps", type=float, default=30.0)
    args = ap.parse_args()

    if args.clip_length != 128:
        raise ValueError("The paper configuration uses 128 frames; keep --clip-length 128")
    args.output.mkdir(parents=True, exist_ok=True)

    json_path, image_meta, full_meta = load_json(args.record)
    image_paths, frames = read_frames(args.record)
    image_ts = timestamp_array(image_meta, len(frames), args.fallback_fps)
    frame_dt = np.diff(image_ts).astype(np.float64) / 1e9
    fps = 1.0 / float(np.median(frame_dt)) if len(frame_dt) else args.fallback_fps

    install_cpu_mamba_stub()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from neural_methods.model.FreqSepPhys import FreqSepPhys

    model = FreqSepPhys(frames=args.clip_length)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {str(k).removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) != 0:
        raise RuntimeError(f"Checkpoint did not load completely; missing keys: {missing[:5]}")
    model.eval()

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    previous = None
    boxes = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if previous is None:
            found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(found) == 0:
                raise RuntimeError("No face detected in the first frame")
            previous = max(found, key=lambda b: int(b[2]) * int(b[3])).astype(int)
        else:
            previous = detect_box(gray, previous, cascade)
        boxes.append(previous.copy())
    boxes = np.asarray(boxes, dtype=np.int32)

    manifest_rows = []
    clip_count = 0
    for start in range(0, len(frames) - args.clip_length + 1, args.stride):
        if args.max_clips and clip_count >= args.max_clips:
            break
        stop = start + args.clip_length
        clip_frames = np.asarray([crop_with_box(frames[i], boxes[i]) for i in range(start, stop)])
        parts = get_decomposition(model, clip_frames)
        clip_id = f"clip_{clip_count:04d}"
        np.savez_compressed(
            args.output / f"{clip_id}.npz",
            **parts,
            slow_fs=np.float32(fps / 4.0),
            fast_fs=np.float32(fps / 2.0),
        )
        bvp_count = write_bvp(
            args.output / f"{clip_id}_bvp.csv",
            full_meta,
            int(image_ts[start]),
            int(image_ts[stop - 1]),
        )
        motion_path = args.output / f"{clip_id}_motion.csv"
        with motion_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ns", "x", "y", "width", "height", "center_x", "center_y", "dx", "dy", "motion_proxy"])
            for j, i in enumerate(range(start, stop)):
                x, y, ww, hh = boxes[i]
                cx, cy = x + ww / 2.0, y + hh / 2.0
                if j == 0:
                    dx = dy = 0.0
                else:
                    px, py, pww, phh = boxes[i - 1]
                    dx = cx - (px + pww / 2.0)
                    dy = cy - (py + phh / 2.0)
                w.writerow([int(image_ts[i]), int(x), int(y), int(ww), int(hh), cx, cy, dx, dy, float(np.hypot(dx, dy))])
        manifest_rows.append({
            "dataset": args.dataset,
            "subject": args.subject,
            "record": args.record_id,
            "condition": args.condition,
            "clip": clip_id,
            "feature_file": f"{clip_id}.npz",
            "bvp_file": f"{clip_id}_bvp.csv",
            "motion_file": f"{clip_id}_motion.csv",
            "frame_start": start,
            "frame_count": args.clip_length,
            "frame_rate_hz": fps,
            "slow_fs_hz": fps / 4.0,
            "fast_fs_hz": fps / 2.0,
            "bvp_samples": bvp_count,
            "checkpoint": str(args.checkpoint),
            "source_json": str(json_path),
        })
        clip_count += 1

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"Exported {clip_count} clips to {args.output}")
    print(f"Source frames: {len(image_paths)}, frame rate: {fps:.4f} Hz")
    print("The NPZ files contain the first slow/fast decomposition before Mamba; they are suitable for reviewer comment 3 frequency-energy analysis.")


if __name__ == "__main__":
    main()
