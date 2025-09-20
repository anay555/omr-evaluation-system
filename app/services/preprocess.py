import cv2
import numpy as np
from typing import Tuple


def detect_orientation(img: np.ndarray) -> Tuple[np.ndarray, int]:
    """Detect coarse orientation in multiples of 90 degrees using edge density.
    Returns rotated_image, rotation_degrees.
    """
    candidates = [0, 90, 180, 270]
    scores = []
    for deg in candidates:
        rot = rotate_image(img, deg)
        gray = cv2.cvtColor(rot, cv2.COLOR_RGB2GRAY) if rot.ndim == 3 else rot
        edges = cv2.Canny(gray, 60, 180)
        # Prefer orientation where horizontal lines are stronger (typical OMR layout)
        sobelx = cv2.Sobel(edges, cv2.CV_32F, 1, 0, ksize=3)
        score = float(np.mean(np.abs(sobelx)))
        scores.append((score, deg))
    best_deg = max(scores, key=lambda x: x[0])[1]
    return rotate_image(img, best_deg), best_deg


def rotate_image(img: np.ndarray, degrees: int) -> np.ndarray:
    if degrees % 360 == 0:
        return img.copy()
    rot_map = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    code = rot_map.get(degrees % 360)
    if code is None:
        # Arbitrary angle
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, degrees, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.rotate(img, code)


def rectify_perspective(img: np.ndarray) -> np.ndarray:
    """Attempt a simple perspective rectification by detecting the largest contour
    and warping to a rectangle. Returns the warped image or the original on failure.
    """
    original = img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
    th = 255 - th
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return original
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        return original
    pts = approx.reshape(4, 2).astype(np.float32)
    # Order points (tl, tr, br, bl)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    warped = cv2.warpPerspective(original, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped
