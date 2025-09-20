from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
from app.core.config import settings

# -----------------------------
# Placeholder scoring utilities
# -----------------------------

def _score_from_darkness(img: np.ndarray) -> float:
    """Very simple proxy scoring based on proportion of dark pixels.
    Returns a value in [0, 100].
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    norm = gray.astype(np.float32) / 255.0
    darkness = 1.0 - norm
    darkness = np.clip(darkness, 0.0, 1.0)
    score = float(darkness.mean() * 100.0)
    return max(0.0, min(100.0, score))


def evaluate_image(img: np.ndarray, sheet_version: str) -> Dict[str, Any]:
    """Aggregate-only placeholder, kept for API compatibility."""
    if sheet_version not in settings.sheet_versions:
        sheet_versions = ", ".join(settings.sheet_versions)
        raise ValueError(f"Unsupported sheet_version. Expected one of: {sheet_versions}")

    total_score = _score_from_darkness(img)
    n = len(settings.subjects)
    per_subject_raw = total_score / n
    per_subject_scores = []
    for s in settings.subjects:
        per_score = min(
            settings.per_subject_max,
            per_subject_raw * (settings.per_subject_max / (settings.total_max / n)),
        )
        per_subject_scores.append({"subject": s, "score": round(per_score, 2)})

    total = sum(x["score"] for x in per_subject_scores)
    total = min(settings.total_max, round(total, 2))

    return {
        "sheet_version": sheet_version,
        "per_subject": per_subject_scores,
        "total": total,
    }

# -----------------------------
# Per-question placeholder logic
# -----------------------------

OPTIONS = ["a", "b", "c", "d"]


def predict_answers(img: np.ndarray, num_questions: int = 100) -> List[str]:
    """Return predicted options for Q1..num_questions using a deterministic
    placeholder based on image statistics. Replace with real OMR.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    h, w = gray.shape
    # Sample a few regions to derive a small feature vector
    samples = [
        gray[h // 4 : h // 4 * 3, w // 4 : w // 4 * 3].mean(),  # center
        gray[: h // 2, : w // 2].mean(),  # tl
        gray[: h // 2, w // 2 :].mean(),  # tr
        gray[h // 2 :, : w // 2].mean(),  # bl
        gray[h // 2 :, w // 2 :].mean(),  # br
    ]
    base = int(sum(int(s) for s in samples) // max(1, len(samples)))

    answers = []
    for i in range(1, num_questions + 1):
        idx = (base + i * 7) % 4  # pseudo hashing by question index
        answers.append(OPTIONS[idx])
    return answers


def subject_for_question(q: int) -> str:
    if 1 <= q <= 20:
        return "Python"
    if 21 <= q <= 40:
        return "EDA"
    if 41 <= q <= 60:
        return "SQL"
    if 61 <= q <= 80:
        return "POWER BI"
    return "Statistics"  # 81..100


def compute_scores_from_answers(answers: List[str], key_map: Dict[int, str]) -> Tuple[Dict[str, int], int]:
    """Compute per-subject and total scores given predicted answers and key."""
    per_subject = {s: 0 for s in ["Python", "EDA", "SQL", "POWER BI", "Statistics"]}
    total = 0
    for q, pred in enumerate(answers, start=1):
        correct = key_map.get(q)
        if correct is not None and str(pred).lower() == str(correct).lower():
            per_subject[subject_for_question(q)] += 1
            total += 1
    return per_subject, total


def format_answers_as_columns(answers: List[str]) -> Dict[str, List[str]]:
    """Produce a 5-column layout like the provided template with strings 'n - x'."""
    columns: Dict[str, List[str]] = {s: [] for s in ["Python", "EDA", "SQL", "POWER BI", "Statistics"]}
    for q, ans in enumerate(answers, start=1):
        s = subject_for_question(q)
        columns[s].append(f"{q} - {ans}")
    # Ensure 20 rows per subject
    for s in columns:
        if len(columns[s]) < 20:
            columns[s] += [""] * (20 - len(columns[s]))
    return columns
