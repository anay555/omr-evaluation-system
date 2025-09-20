from typing import Dict
import re
import pandas as pd
from io import BytesIO

SUBJECT_RANGES = {
    "Python": (1, 20),
    "EDA": (21, 40),
    "SQL": (41, 60),
    "POWER BI": (61, 80),
    "Statistics": (81, 100),
}

# Accept common header variants/typos (normalized: lowercase letters only)
SUBJECT_ALIASES = {
    "python": {"python"},
    "eda": {"eda"},
    "sql": {"sql"},
    "powerbi": {"powerbi", "powerb i", "pbi"},
    "statistics": {"statistics", "stat", "stats", "statisitcs", "satisitcs", "statisics"},
}

LINE_RE = re.compile(r"^\s*(\d+)\s*[-–]\s*([a-dA-D])")


def _norm(s: object) -> str:
    return re.sub(r"[^a-z]", "", str(s).strip().lower())


def _match_column(columns, subject: str):
    target_norm = _norm(subject)
    aliases = SUBJECT_ALIASES.get(target_norm, {target_norm})
    for c in columns:
        if _norm(c) in aliases:
            return c
    return None


def parse_key_dataframe(df: pd.DataFrame) -> Dict[int, str]:
    """Parse a dataframe with columns named like the subjects and cells like 'n - a'.
    Returns {question_number: correct_option_letter}.
    """
    key: Dict[int, str] = {}
    for subject, (start, end) in SUBJECT_RANGES.items():
        col = subject if subject in df.columns else _match_column(df.columns, subject)
        if col is None:
            continue
        col_series = df[col].dropna().astype(str)
        for cell in col_series:
            m = LINE_RE.match(cell.strip())
            if not m:
                continue
            q = int(m.group(1))
            ans = m.group(2).lower()
            if start <= q <= end:
                key[q] = ans
    return key


def parse_key_excel(file_obj_or_bytes, sheet_name: str) -> Dict[int, str]:
    data = file_obj_or_bytes
    if isinstance(file_obj_or_bytes, (bytes, bytearray)):
        data = BytesIO(file_obj_or_bytes)
    xls = pd.ExcelFile(data)
    sheet = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    return parse_key_dataframe(df)
