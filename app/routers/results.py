from fastapi import APIRouter, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db.crud import get_db, upsert_student, create_evaluation, list_evaluations

router = APIRouter(prefix="/results", tags=["results"]) 

class EvalIn(BaseModel):
    student_code: str
    sheet_version: str
    per_subject: Dict[str, float]
    total: float
    details: Optional[Dict[str, Any]] = None

@router.post("/")
def create_result(payload: EvalIn, db: Session = Depends(get_db)):
    upsert_student(db, payload.student_code)
    ev = create_evaluation(db, payload.student_code, payload.sheet_version,
                           payload.per_subject, payload.total, payload.details)
    return {"id": ev.id}

@router.get("/")
def list_results(db: Session = Depends(get_db)):
    rows = list_evaluations(db)
    return rows
