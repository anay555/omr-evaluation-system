from typing import Dict, Any, List
import json
import numpy as np
import cv2


def _clahe_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _roi_from_norm(norm_roi: List[float], w: int, h: int) -> slice:
    x0, y0, x1, y1 = norm_roi
    x0i, y0i = int(max(0, min(w - 1, x0 * w))), int(max(0, min(h - 1, y0 * h)))
    x1i, y1i = int(max(0, min(w, x1 * w))), int(max(0, min(h, y1 * h)))
    return slice(y0i, y1i), slice(x0i, x1i)


def _fill_ratio(gray: np.ndarray, roi: slice) -> float:
    ysl, xsl = roi
    patch = gray[ysl, xsl]
    if patch.size == 0:
        return 0.0
    # Adaptive threshold then compute dark pixel ratio
    th = cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 5)
    dark = th > 0
    return float(dark.mean())


def _apply_adjust(norm_roi: List[float], sx: float, sy: float, ox: float, oy: float) -> List[float]:
    x0, y0, x1, y1 = norm_roi
    x0 = min(1.0, max(0.0, x0 * sx + ox))
    x1 = min(1.0, max(0.0, x1 * sx + ox))
    y0 = min(1.0, max(0.0, y0 * sy + oy))
    y1 = min(1.0, max(0.0, y1 * sy + oy))
    return [x0, y0, x1, y1]


def evaluate_with_template(img: np.ndarray, template: Dict[str, Any],
                           fill_threshold: float = 0.45,
                           min_margin: float = 0.12,
                           scale_x: float = 1.0,
                           scale_y: float = 1.0,
                           offset_x: float = 0.0,
                           offset_y: float = 0.0) -> List[str]:
    """
    Evaluate answers using a JSON template with normalized ROIs.
    - fill_threshold: minimum dark ratio to consider a bubble filled
    - min_margin: winner's fill minus next best must exceed this margin, else mark blank
    - scale_x/scale_y & offset_x/offset_y: fine adjustments to align ROIs to the image
    Returns list of answers for questions sorted by index.
    """
    gray = _clahe_gray(img)
    h, w = gray.shape

    questions = sorted(template.get("questions", []), key=lambda q: q.get("index", 0))
    answers: List[str] = []
    for q in questions:
        opts = q.get("options", {})
        scores = {}
        for opt, norm_roi in opts.items():
            adj = _apply_adjust(norm_roi, scale_x, scale_y, offset_x, offset_y)
            roi = _roi_from_norm(adj, w, h)
            scores[opt] = _fill_ratio(gray, roi)
        if not scores:
            answers.append("")
            continue
        best = max(scores, key=scores.get)
        s_best = scores[best]
        s_second = max([v for k, v in scores.items() if k != best] or [0.0])
        if s_best >= fill_threshold and (s_best - s_second) >= min_margin:
            answers.append(best)
        else:
            answers.append("")  # ambiguous or blank
    return answers


def draw_overlay(img: np.ndarray, template: Dict[str, Any], answers: List[str],
                 scale_x: float = 1.0, scale_y: float = 1.0,
                 offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
    """Draw rectangles for each option; green for selected, red for others."""
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    h, w = vis.shape[:2]
    questions = sorted(template.get("questions", []), key=lambda q: q.get("index", 0))
    for idx, q in enumerate(questions, start=1):
        opts = q.get("options", {})
        selected = answers[idx - 1] if idx - 1 < len(answers) else ""
        for opt, norm_roi in opts.items():
            adj = _apply_adjust(norm_roi, scale_x, scale_y, offset_x, offset_y)
            ysl, xsl = _roi_from_norm(adj, w, h)
            x0, y0 = xsl.start, ysl.start
            x1, y1 = xsl.stop, ysl.stop
            color = (0, 200, 0) if opt == selected else (200, 50, 50)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
    return vis
