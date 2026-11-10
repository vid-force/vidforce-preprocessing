"""Depth Anything 3 wrapper: metric depth and intrinsics of one frame (``metric_depth``) or of a stack of frames
stored at a reduced resolution (``da3_depth_stack``). The package ``depth_anything_3`` must be importable; set
``DEPTH_ANYTHING_3_SRC`` to a checkout of its repository otherwise.
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Tuple

import numpy as np
import torch

_DA3_SRC = os.environ.get("DEPTH_ANYTHING_3_SRC")
MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
_PROCESS_RES = 504

_DA3 = None


def _ensure_paths() -> None:
    if not _DA3_SRC:
        return
    for p in (pathlib.Path(_DA3_SRC), pathlib.Path(_DA3_SRC) / "src"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _load_model(device: torch.device):
    global _DA3
    if _DA3 is not None:
        return _DA3
    _ensure_paths()
    from depth_anything_3.api import DepthAnything3

    print(f"[depth] loading {MODEL_ID} on {device} (one-time)", flush=True)
    _DA3 = DepthAnything3.from_pretrained(MODEL_ID).to(device).eval()
    return _DA3


def metric_depth(
    rgb: np.ndarray, device: str | torch.device = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be (H, W, 3) uint8; got {rgb.shape} {rgb.dtype}")
    H, W = rgb.shape[:2]
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    model = _load_model(dev)
    with torch.no_grad():
        pred = model.inference(image=[rgb], process_res=_PROCESS_RES)
    depth_m = np.asarray(pred.depth, dtype=np.float32)
    if depth_m.ndim == 3:
        depth_m = depth_m[0]
    K = np.asarray(pred.intrinsics, dtype=np.float64)
    if K.ndim == 3:
        K = K[0]
    if depth_m.shape != (H, W):
        import cv2

        sy, sx = H / depth_m.shape[0], W / depth_m.shape[1]
        depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_LINEAR)
        K = K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
    return (depth_m * 1000.0).astype(np.float32), K


def da3_depth_stack(frames_rgb: np.ndarray, device: str | torch.device = "cuda",
                    store_long_side: int = 160):
    import cv2

    frames_rgb = np.asarray(frames_rgb)
    if frames_rgb.ndim != 4 or frames_rgb.dtype != np.uint8:
        raise ValueError(f"frames_rgb must be (K,H,W,3) uint8, got {frames_rgb.shape} {frames_rgb.dtype}")
    maps, ks = [], []
    for frame in frames_rgb:
        depth_mm, K_full = metric_depth(frame, device=device)
        h, w = depth_mm.shape
        scale = max(h, w) / float(store_long_side)
        dh, dw = max(8, int(round(h / scale))), max(8, int(round(w / scale)))
        d = cv2.resize(depth_mm, (dw, dh), interpolation=cv2.INTER_AREA)
        K = K_full.copy()
        K[0, 0] *= dw / w
        K[0, 2] *= dw / w
        K[1, 1] *= dh / h
        K[1, 2] *= dh / h
        maps.append((d / 1000.0).astype(np.float16))
        ks.append(K)
    return np.stack(maps), np.stack(ks)
