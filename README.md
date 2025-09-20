# OMR Evaluation System

A scalable, automated OMR (Optical Mark Recognition) evaluation system that:

- Accurately evaluates OMR sheets captured via mobile phone camera
- Provides per-subject scores (0–20 each) and a total score (0–100)
- Works with multiple sheet versions (2–4 sets per exam)
- Functions online via a web application interface for evaluators to manage results
- Ensures <0.5% error tolerance, aligned with Innomatics’ quality standards
- Reduces evaluation turnaround from days to minutes, freeing evaluators to focus on insights and student engagement

## Tech Stack

- Backend: Python, FastAPI
- Image Processing: OpenCV, NumPy, scikit-image, Pillow
- Packaging/Env: venv, pip
- Testing: pytest

## High-Level Architecture

- Ingestion (Web): Upload photo(s) of OMR sheets (mobile capture supported)
- Preprocessing: Perspective correction, illumination normalization, denoising, binarization
- Alignment: Template-based alignment for different sheet versions (2–4 sets)
- Bubble Detection & Classification: Adaptive thresholding + contour analysis (and/or Hough), resolve multi-marks
- Scoring: Per-subject aggregation (0–20), total score (0–100), configurable answer keys per sheet version
- API/Storage: REST endpoints to submit sheets, retrieve results; persistence layer can be added (e.g., Postgres)
- Admin UI: Simplified web interface for evaluators to manage results

## Quality Targets

- < 0.5% error tolerance across end-to-end evaluation, validated on a holdout dataset
- Robust to common capture issues: skew, shadows, slight blur
- Deterministic and auditable scoring logic with trace artifacts for QA

## Project Layout (initial)

- app/
  - main.py (FastAPI app entry)
  - routers/
    - evaluate.py (upload & evaluate endpoints)
  - services/
    - omr.py (OMR processing pipeline stub)
  - core/
    - config.py (settings and constants)
- tests/
  - test_app.py (basic health tests)
- requirements.txt
- README.md
- .gitignore

## Running Locally

1) Create and activate virtual environment

   Windows (PowerShell):
   - python -m venv .venv
   - .venv\\Scripts\\Activate.ps1

   macOS/Linux:
   - python3 -m venv .venv
   - source .venv/bin/activate

2) Install dependencies

   - pip install -r requirements.txt

3) Run the API server (FastAPI)

   - uvicorn app.main:app --reload --port 8000

   Then open http://localhost:8000/docs to explore the API.

4) Run the frontend (Streamlit)

   - python -m streamlit run streamlit_app.py

   The Streamlit app uses the internal OMR service directly for local evaluation.

