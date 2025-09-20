# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- Purpose: Evaluate OMR (Optical Mark Recognition) sheets with a FastAPI backend and a Streamlit UI. Includes a basic per-subject scoring layout and optional template-driven ROI detection.
- Languages/stack: Python 3, FastAPI, Uvicorn, NumPy/OpenCV/scikit-image, Streamlit, SQLite (SQLAlchemy), pytest.

Common commands
- Setup (virtual environment and deps)
  - Windows PowerShell
    - python -m venv .venv
    - .venv\Scripts\Activate.ps1
    - pip install -r requirements.txt
  - macOS/Linux
    - python3 -m venv .venv
    - source .venv/bin/activate
    - pip install -r requirements.txt

- Run API (FastAPI)
  - uvicorn app.main:app --reload --port 8000
  - API docs: http://localhost:8000/docs

- Run Streamlit UI
  - python -m streamlit run streamlit_app.py

- Tests (pytest)
  - Run all tests: pytest -q
  - Run a single test: pytest tests/test_app.py::test_health -q

- Quick API checks
  - Health: curl http://localhost:8000/health
  - Evaluate (multipart form upload):
    - curl -X POST -F "sheet_version=A" -F "file=@/path/to/image.jpg" http://localhost:8000/api/evaluate

High-level architecture
- FastAPI application (app/main.py)
  - Mounts two routers under /api
    - /api/evaluate (app/routers/evaluate.py)
      - Accepts multipart image + sheet_version
      - Converts to NumPy image and calls app.services.omr.evaluate_image
    - /api/results (app/routers/results.py)
      - Persists evaluation summaries to SQLite via app.db.*
      - Endpoints:
        - POST /api/results/ accepts { student_code, sheet_version, per_subject, total, details? }
        - GET /api/results/ lists recent evaluations

- Services (app/services)
  - omr.py
    - Placeholder aggregate scoring based on image darkness
    - Per-question pipeline stubs:
      - predict_answers: generates deterministic answers per image
      - compute_scores_from_answers: maps answers to per-subject and total via a provided key
      - format_answers_as_columns: formats answers into 5 subject columns (20 each) for export
  - preprocess.py
    - detect_orientation: coarse 0/90/180/270 rotation selection
    - rectify_perspective: largest-contour warp to rectangular sheet
  - omr_template.py
    - Template-driven ROI evaluation using normalized [x0,y0,x1,y1] coordinates
    - Supports fill_threshold/min_margin and global scale/offset adjustments
    - draw_overlay helper for ROI visualization
  - grid.py
    - Naive grid ROI generator (evenly spaced, 100×4)
  - key.py
    - Parses answer keys from Excel (openpyxl/pandas) with robust column matching and "n - x" cell parsing

- Data & persistence (app/db)
  - SQLite at sqlite:///./omr.db (created on import)
  - models.py defines Student and Evaluation tables; Base.metadata.create_all() runs at module import
  - crud.py exposes Session management, upsert_student, create_evaluation, list_evaluations, summary_by_subject

- Configuration (app/core/config.py)
  - settings: subjects, per-subject max, total max, and supported sheet versions

- Streamlit application (streamlit_app.py)
  - End-to-end local workflow for evaluators: upload images (up to 20), optional template JSON, optional Excel answer key
  - Uses services for preprocessing, answer prediction or template-based selection, scoring, charts, and export (Excel/CSV)
  - Optional API persistence: posts to POST /api/results/ if the FastAPI server is running and "Save to DB" is enabled

- Templates (templates/example_template.json)
  - Example normalized ROI layout compatible with omr_template.evaluate_with_template

Important notes from README
- Install dependencies with pip install -r requirements.txt
- Run API with uvicorn app.main:app --reload --port 8000 and use /docs for interactive testing
- Streamlit UI entry: python -m streamlit run streamlit_app.py

Development tips specific to this repo
- When using the template-driven flow, you can tweak scale_x/scale_y and offset_x/offset_y in the UI to align ROIs without regenerating the JSON
- If you’re only testing the API health, the minimal test is tests/test_app.py; expand tests under tests/ as you implement more of the OMR pipeline
