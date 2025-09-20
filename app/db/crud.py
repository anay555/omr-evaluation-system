from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from .models import SessionLocal, Student, Evaluation


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def upsert_student(db: Session, student_code: str) -> Student:
    obj = db.query(Student).filter(Student.student_code == student_code).first()
    if not obj:
        obj = Student(student_code=student_code)
        db.add(obj)
        db.commit()
        db.refresh(obj)
    return obj


def create_evaluation(db: Session, student_code: str, sheet_version: str,
                      per_subject: Dict[str, float], total: float,
                      details: Optional[Dict[str, Any]] = None) -> Evaluation:
    ev = Evaluation(
        student_code=student_code,
        sheet_version=sheet_version,
        per_subject=per_subject,
        total=total,
        details=details or {},
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev


def list_evaluations(db: Session, limit: int = 100) -> List[Evaluation]:
    return db.query(Evaluation).order_by(Evaluation.id.desc()).limit(limit).all()


def summary_by_subject(db: Session) -> Dict[str, Any]:
    rows = db.query(Evaluation).all()
    agg: Dict[str, Dict[str, float]] = {}
    for r in rows:
        for s, v in (r.per_subject or {}).items():
            agg.setdefault(s, {"sum": 0.0, "count": 0}).
            agg[s]["sum"] += float(v)
            agg[s]["count"] += 1
    out = {k: (v["sum"]/max(1, v["count"])) for k, v in agg.items()}
    return out
