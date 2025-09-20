from pydantic import BaseModel
from typing import List

class Settings(BaseModel):
    subjects: List[str] = [
        "Python",
        "EDA",
        "SQL",
        "POWER BI",
        "Statistics",
    ]
    per_subject_max: int = 20
    total_max: int = 100
    sheet_versions: List[str] = ["A", "B", "C", "D"]

settings = Settings()
