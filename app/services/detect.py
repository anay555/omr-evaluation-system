from typing import Dict, Any, List, Tuple
import numpy as np
import cv2

# Threshold utilities adapted from proven OMR approaches (re-implemented)
# We operate on mean intensities inside each option ROI and pick selections
# by finding large gaps in distributions (global + local thresholds).


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    return gray


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _roi_from_norm(norm_roi: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = norm_roi
    x0i, y0i = int(max(0, min(w - 1, x0 * w))), int(max(0, min(h - 1, y0 * h)))
    x1i, y1i = int(max(0, min(w, x1 * w))), int(max(0, min(h, y1 * h)))
    return x0i, y0i, x1i, y1i


def _mean_intensity(gray: np.ndarray, roi_box: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = roi_box
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 255.0
    return float(np.mean(patch))


def _largest_gap_threshold(vals: List[float], looseness: int = 1, min_jump: float = 8.0, default_thr: float = 160.0) -> float:
    if not vals:
        return default_thr
    vs = sorted(vals)
    ls = max(1, (looseness + 1) // 2)
    l = len(vs) - ls
    max_jump = min_jump
    thr = default_thr
    for i in range(ls, l):
        jump = vs[i + ls] - vs[i - ls]
        if jump > max_jump:
            max_jump = jump
            thr = vs[i - ls] + jump / 2.0
    return float(thr)


def _choose_option_by_threshold(intensities: Dict[str, float], local_thr: float, margin: float = 6.0) -> str:
    # Lower intensity => darker => marked. Select those below local_thr.
    below = {k: v for k, v in intensities.items() if v <= local_thr}
    if not below:
        return ""  # blank/uncertain
    # Pick the lowest intensity (darkest)
    best_opt, best_val = min(below.items(), key=lambda kv: kv[1])
    # Disambiguate with margin from second-best (if any)
    others = [v for k, v in below.items() if k != best_opt]
    if others:
        second = min(others)
        if (second - best_val) < margin:
            return ""  # ambiguous
    return best_opt


def evaluate_by_questions(img: np.ndarray, questions: List[Dict[str, Any]],
                          scale_x: float = 1.0, scale_y: float = 1.0,
                          offset_x: float = 0.0, offset_y: float = 0.0) -> List[str]:
    """
    Evaluate answers given a list of question dicts with 'index' and 'options' -> normalized ROIs.
    Returns list of selected options or empty string for blank/ambiguous.
    """
    gray = _clahe(_to_gray(img))
    h, w = gray.shape

    # Collect all intensities across all options for global calibration
    all_vals: List[float] = []
    per_question_vals: List[Tuple[int, Dict[str, float]]] = []

    def _adj(norm_roi: List[float]) -> List[float]:
        x0, y0, x1, y1 = norm_roi
        return [x0 * scale_x + offset_x, y0 * scale_y + offset_y, x1 * scale_x + offset_x, y1 * scale_y + offset_y]

    for q in sorted(questions, key=lambda qq: qq.get("index", 0)):
        opts = q.get("options", {})
        intensities: Dict[str, float] = {}
        for opt, norm_roi in opts.items():
            box = _roi_from_norm(_adj(norm_roi), w, h)
            intensities[opt] = _mean_intensity(gray, box)
            all_vals.append(intensities[opt])
        per_question_vals.append((q.get("index", 0), intensities))

    global_thr = _largest_gap_threshold(all_vals, looseness=3, min_jump=6.0, default_thr=160.0)

    answers: List[str] = []
    for _, intensities in per_question_vals:
        local_thr = _largest_gap_threshold(list(intensities.values()), looseness=1, min_jump=4.0, default_thr=global_thr)
        ans = _choose_option_by_threshold(intensities, local_thr, margin=6.0)
        answers.append(ans)
    return answers