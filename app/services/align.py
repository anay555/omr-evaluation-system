from typing import Dict, Any, List, Tuple
import random
import numpy as np
import cv2


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _vertical_edges(gray: np.ndarray) -> np.ndarray:
    # Emphasize vertical structures using Sobel X
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    mag = np.abs(sobelx)
    mag = mag / (mag.max() + 1e-6)
    return (mag * 255.0).astype(np.uint8)


def _roi_from_norm(norm_roi: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = norm_roi
    x0i, y0i = int(max(0, min(w - 1, x0 * w))), int(max(0, min(h - 1, y0 * h)))
    x1i, y1i = int(max(0, min(w, x1 * w))), int(max(0, min(h, y1 * h)))
    return x0i, y0i, x1i, y1i


def _sample_option_rois(template: Dict[str, Any], max_rois: int = 200) -> List[List[float]]:
    qs = sorted(template.get("questions", []), key=lambda q: q.get("index", 0))
    rois: List[List[float]] = []
    for q in qs:
        opts = q.get("options", {})
        for _, roi in opts.items():
            rois.append(list(roi))
    if len(rois) > max_rois:
        random.seed(42)
        rois = random.sample(rois, max_rois)
    return rois


def estimate_offset(img: np.ndarray, template: Dict[str, Any],
                    search_px_ratio: float = 0.02, steps: int = 21) -> Tuple[float, float]:
    """
    Estimate a small horizontal/vertical offset (normalized 0..1) to better align
    template ROIs to the image by maximizing vertical edge response inside ROIs.
    Returns (offset_x, offset_y) in normalized coordinates.
    """
    gray = _clahe(_to_gray(img))
    edges = _vertical_edges(gray)
    h, w = edges.shape

    rois = _sample_option_rois(template, max_rois=200)
    if not rois:
        return 0.0, 0.0

    # Search small offsets around zero
    max_off = search_px_ratio
    xs = np.linspace(-max_off, max_off, steps)
    ys = np.linspace(-max_off, max_off, 1)  # keep vertical fixed for now (performance)

    best_score = -1.0
    best = (0.0, 0.0)

    for dx in xs:
        for dy in ys:
            s = 0.0
            cnt = 0
            for r in rois:
                x0, y0, x1, y1 = r
                x0a = min(1.0, max(0.0, x0 + dx))
                x1a = min(1.0, max(0.0, x1 + dx))
                y0a = min(1.0, max(0.0, y0 + dy))
                y1a = min(1.0, max(0.0, y1 + dy))
                x0i, y0i, x1i, y1i = _roi_from_norm([x0a, y0a, x1a, y1a], w, h)
                if x1i > x0i and y1i > y0i:
                    patch = edges[y0i:y1i, x0i:x1i]
                    if patch.size:
                        s += float(np.mean(patch))
                        cnt += 1
            if cnt:
                score = s / cnt
                if score > best_score:
                    best_score = score
                    best = (float(dx), float(dy))
    return best