from typing import List, Dict, Any

# Placeholder grid detection. In production, you would detect grid corners and
# derive ROIs. We keep it as a thin wrapper so callers can switch between a
# detected grid and a provided template.

def estimate_grid_rois(image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """
    Return a list of question dicts with 'index' and 'options' -> normalized ROIs.
    This is a naive evenly-spaced grid for 100 questions x 4 options.
    """
    cols_per_option = 4
    questions = []
    # Five blocks (20 questions each) stacked vertically
    for q in range(1, 101):
        block = (q - 1) // 20  # 0..4
        row_in_block = (q - 1) % 20
        y_top = 0.08 + block * 0.18 + row_in_block * 0.012
        y_bottom = y_top + 0.012
        x0 = 0.10
        options = {}
        step = 0.12
        for i, opt in enumerate(["a", "b", "c", "d"]):
            x1 = x0 + step
            options[opt] = [x0, y_top, x1, y_bottom]
            x0 = x1 + 0.02
        questions.append({"index": q, "options": options})
    return questions
