from typing import Optional, List, Dict, Any
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./omr.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    student_code = Column(String, unique=True, index=True)  # filename or provided code

class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True, index=True)
    student_code = Column(String, index=True)
    sheet_version = Column(String, index=True)
    per_subject = Column(JSON)
    total = Column(Float)
    details = Column(JSON)  # optional: per-question answers
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)
