import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import pandas as pd
import os

from app.core.config import settings
from app.services.omr import evaluate_image, predict_answers, compute_scores_from_answers, format_answers_as_columns
from app.services.key import parse_key_excel
from app.services.omr_template import evaluate_with_template

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# -------------- UI helpers --------------
def _inject_css(theme: str = "Dark"):
    theme = (theme or "Dark").lower()
    if theme == "light":
        css_vars = """
        :root {
          --brand-primary: #4F46E5; /* indigo-600 */
          --brand-accent: #0EA5E9;  /* sky-500 */
          --bg-grad-start: #f8fafc; /* slate-50 */
          --bg-grad-end: #eef2ff;   /* indigo-50 */
          --card-bg: rgba(255,255,255,0.75);
          --card-border: rgba(2,6,23,0.08);
          --muted: #475569;        /* slate-600 */
          --text: #0f172a;         /* slate-900 */
          --radius: 14px;
        }
        """
    else:
        css_vars = """
        :root {
          --brand-primary: #6366F1; /* indigo-500 */
          --brand-accent: #22D3EE;  /* cyan-400 */
          --bg-grad-start: #0f172a; /* slate-900 */
          --bg-grad-end: #111827;   /* gray-900 */
          --card-bg: rgba(17,24,39,0.55);
          --card-border: rgba(255,255,255,0.08);
          --muted: #94a3b8;
          --text: #E5E7EB;
          --radius: 14px;
        }
        """

    css_rest = """
        /* App background */
        .main > div {
          background: radial-gradient(60% 90% at 10% 0%, rgba(79,70,229,0.08) 0%, rgba(79,70,229,0.0) 60%),
                      linear-gradient(180deg, var(--bg-grad-start), var(--bg-grad-end));
          color: var(--text);
          padding-bottom: 2rem;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {background: #0b1220; color: var(--text);}        
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {color: #d1d5db;}
        [data-testid="stSidebar"] .stSlider, [data-testid="stSidebar"] .stCheckbox {filter: drop-shadow(0 0 0 transparent);}        
        /* Hero */
        .hero {
          background: radial-gradient(80% 80% at 20% 10%, rgba(79,70,229,0.25) 0%, rgba(79,70,229,0.0) 60%),
                      linear-gradient(90deg, rgba(34,211,238,0.15), rgba(79,70,229,0.15));
          border: 1px solid var(--card-border);
          border-radius: 18px; padding: 26px 30px; margin: 12px 0 10px 0;
          box-shadow: 0 12px 28px rgba(2,6,23,0.45);
        }
        .hero h1 {margin: 0 0 8px 0; font-size: 30px; color: #fff; letter-spacing: 0.2px;}
        .hero p {margin: 0; color: #cbd5e1;}
        /* Buttons */
        .stButton>button {
          background: linear-gradient(90deg, var(--brand-primary), var(--brand-accent));
          color: white; border: 0; padding: 0.65rem 1.05rem; border-radius: 12px;
          box-shadow: 0 10px 30px rgba(79,70,229,0.35); transition: transform .06s ease, box-shadow .2s ease;
        }
        .stButton>button:hover {box-shadow: 0 12px 36px rgba(79,70,229,0.45);}        
        .stButton>button:active {transform: translateY(1px);}        
        /* Upload dropzone */
        [data-testid="stFileUploaderDropzone"] {
          background: var(--card-bg);
          border: 1px dashed var(--card-border);
          border-radius: var(--radius);
        }
        /* Tabs */
        [data-baseweb="tab-list"] {
          background: var(--card-bg);
          border: 1px solid var(--card-border);
          border-radius: var(--radius);
        }
        /* Metrics */
        [data-testid="stMetric"] {
          background: var(--card-bg);
          border: 1px solid var(--card-border);
          border-radius: var(--radius);
          padding: 12px 14px;
        }
        [data-testid="stMetricLabel"] {color: var(--muted);}        
        /* Expander */
        .streamlit-expanderHeader {font-weight: 600;}
        /* Dataframe tint */
        .stDataFrame {background: var(--card-bg); border-radius: 12px;}
    """

    st.markdown(f"""
        <style>
        {css_vars}
        {css_rest}
        </style>
    """, unsafe_allow_html=True)


def _status_metrics(uploaded_files, key_map, tpl_file):
    files_count = len(uploaded_files) if uploaded_files else 0
    key_status = "Loaded" if key_map else "Not loaded"
    tpl_status = "Loaded" if tpl_file else "Not loaded"
    c1, c2, c3 = st.columns(3)
    c1.metric("Images selected", files_count)
    c2.metric("Answer key", key_status)
    c3.metric("Template", tpl_status)

st.set_page_config(page_title="OMR Evaluator", page_icon="📝", layout="wide")

# Theme toggle (Light/Dark)
theme_choice = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)
_inject_css(theme_choice)

st.markdown("""
<div class="hero">
  <h1>📝 OMR Evaluation</h1>
  <p>Upload up to 500 images, align with a template, and export results. Accuracy-first, modern UI.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
sheet_version = st.sidebar.selectbox("Sheet Version", settings.sheet_versions, index=0)
st.sidebar.markdown("Max per-subject: **{}** | Total: **{}**".format(settings.per_subject_max, settings.total_max))

# Answer key upload (Excel) — available in both sidebar and main pane
key_file_sidebar = st.sidebar.file_uploader("Upload Answer Key (.xlsx)", type=["xlsx"]) 
key_sheet = None
key_map = None

# Sidebar workflow first
if key_file_sidebar is not None:
    try:
        key_bytes = key_file_sidebar.getvalue()
        xls = pd.ExcelFile(BytesIO(key_bytes))
        key_sheet = st.sidebar.selectbox("Key sheet (Set)", options=xls.sheet_names, index=0)
        key_map = parse_key_excel(key_bytes, key_sheet)
        st.sidebar.success(f"Loaded key from sheet: {key_sheet}")
    except Exception as e:
        st.sidebar.error(f"Failed to read key: {e}")

# Main pane uploader (visible if user misses the sidebar)
with st.expander("Answer Key (Excel)", expanded=key_map is None):
    key_file_main = st.file_uploader("Upload Key (.xlsx)", type=["xlsx"], key="key_main")
    if key_map is None and key_file_main is not None:
        try:
            key_bytes = key_file_main.getvalue()
            xls = pd.ExcelFile(BytesIO(key_bytes))
            key_sheet = st.selectbox("Key sheet (Set)", options=xls.sheet_names, index=0, key="key_sheet_main")
            key_map = parse_key_excel(key_bytes, key_sheet)
            st.success(f"Loaded key from sheet: {key_sheet}")
        except Exception as e:
            st.error(f"Failed to read key: {e}")

if key_map is None:
    st.sidebar.warning("No answer key loaded. Upload the Excel key to compute scores. Without a key, the app will only output predictions and the Summary will not include subject totals.")

# Optional template JSON
with st.expander("Template (JSON) - optional but improves accuracy", expanded=False):
    tpl_file = st.file_uploader("Upload Template JSON", type=["json"], key="tpl_json")
    fill_threshold = st.slider("Fill threshold", 0.1, 0.9, 0.45, 0.01)
    min_margin = st.slider("Min margin between top 2 options", 0.0, 0.5, 0.12, 0.01)
    st.caption("Alignment fine-tuning (applied to all images in this run):")
    scale_x = st.slider("Scale X", 0.9, 1.1, 1.0, 0.005)
    scale_y = st.slider("Scale Y", 0.9, 1.1, 1.0, 0.005)
    offset_x = st.slider("Offset X", -0.05, 0.05, 0.0, 0.001)
    offset_y = st.slider("Offset Y", -0.05, 0.05, 0.0, 0.001)
    show_overlay = st.checkbox("Show debug overlay for first image", value=False)

uploaded_files = st.file_uploader(
    "Select up to 500 OMR images (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

save_to_db = st.checkbox("Save results to local database (SQLite)", value=False)
student_id_hint = st.text_input("Student ID pattern (optional)", value="filename_without_extension")

# Processing options for large batches
large_batch = st.checkbox(
    "Large batch mode (skip per-image sheets/overlay)",
    value=False,
    help="Recommended for 200+ images. Reduces memory and speeds up export by omitting per-image Excel sheets and overlay preview.",
)

if uploaded_files:
    if len(uploaded_files) > 500:
        st.warning("You uploaded more than 500 files; only the first 500 will be evaluated.")
        uploaded_files = uploaded_files[:500]

    _status_metrics(uploaded_files, key_map, tpl_file)

    if st.button(f"Evaluate {len(uploaded_files)} image(s)"):
        results = []
        detailed_sheets = []  # for per-image template-like sheets
        progress = st.progress(0)
        for i, uf in enumerate(uploaded_files, 1):
            try:
                image = Image.open(uf).convert("RGB")
                np_img = np.array(image)

                # Preprocess: orientation & perspective
                from app.services.preprocess import detect_orientation, rectify_perspective
                np_img, _ = detect_orientation(np_img)
                np_img = rectify_perspective(np_img)

                # Per-question predictions
                if tpl_file is not None:
                    import json as _json
                    tpl = _json.loads(tpl_file.getvalue().decode("utf-8"))
                    from app.services.detect import evaluate_by_questions
                    questions = sorted(tpl.get("questions", []), key=lambda q: q.get("index", 0))
                    answers = evaluate_by_questions(
                        np_img,
                        questions,
                        scale_x=scale_x, scale_y=scale_y,
                        offset_x=offset_x, offset_y=offset_y,
                    )
                else:
                    # Template-free: use naive grid and threshold-based detection
                    from app.services.grid import estimate_grid_rois
                    from app.services.detect import evaluate_by_questions
                    h, w = np_img.shape[:2]
                    questions = estimate_grid_rois(w, h)
                    answers = evaluate_by_questions(np_img, questions)

                # Scores (if key provided)
                per_subj_scores = {s: None for s in settings.subjects}
                total_score = None
                if key_map:
                    per_subj_scores, total_score = compute_scores_from_answers(answers, key_map)

                # Row for summary table
                row = {"filename": uf.name, "Set": key_sheet or sheet_version}
                for s in settings.subjects:
                    if per_subj_scores[s] is not None:
                        row[s] = per_subj_scores[s]
                if total_score is not None:
                    row["total"] = total_score
                results.append(row)

                # Prepare per-image sheet data in your template style
                if not large_batch:
                    cols = format_answers_as_columns(answers)
                    detailed_sheets.append((uf.name, pd.DataFrame(cols)))

                # Optionally save to DB via FastAPI
                if save_to_db and key_map:
                    per_subj_scores = {s: row.get(s) for s in settings.subjects if s in row}
                    # Student code from filename (without extension) unless pattern provided
                    import os
                    student_code = os.path.splitext(uf.name)[0]
                    import requests
                    try:
                        payload = {
                            "student_code": student_code,
                            "sheet_version": key_sheet or sheet_version,
                            "per_subject": per_subj_scores,
                            "total": row.get("total", 0),
                            "details": {"answers": answers},
                        }
                        # Call deployed API if configured, else local
                        requests.post(f"{API_BASE_URL}/api/results/", json=payload, timeout=2)
                    except Exception:
                        pass
            except Exception as e:
                results.append({"filename": uf.name, "error": str(e)})
            progress.progress(int(i * 100 / len(uploaded_files)))

        st.success("Evaluation complete!")

        # Debug overlay preview for first image
        if tpl_file is not None and show_overlay and uploaded_files and not large_batch:
            import json as _json
            tpl = _json.loads(tpl_file.getvalue().decode("utf-8"))
            try:
                first_img = Image.open(uploaded_files[0]).convert("RGB")
                np_first = np.array(first_img)
                from app.services.preprocess import detect_orientation, rectify_perspective
                np_first, _ = detect_orientation(np_first)
                np_first = rectify_perspective(np_first)
                from app.services.omr_template import draw_overlay
                overlay = draw_overlay(np_first, tpl, answers,
                                       scale_x=scale_x, scale_y=scale_y,
                                       offset_x=offset_x, offset_y=offset_y)
                st.image(overlay, caption="ROI overlay (first image)", use_container_width=True)
            except Exception as e:
                st.info(f"Could not render overlay: {e}")

        if results:
            df = pd.DataFrame(results)

            tab_sum, tab_charts, tab_dl = st.tabs(["Summary", "Charts", "Downloads"])

            with tab_sum:
                st.dataframe(df, use_container_width=True)

            # Prepare export
            subject_cols = list(settings.subjects)
            ordered_cols = [c for c in ["filename", "Set", *subject_cols, "total"] if c in df.columns]
            df_for_export = df[ordered_cols] if ordered_cols else df

            # Separate errors if any
            if "error" in df.columns:
                err = df[["filename", "error"]].dropna()
                if not err.empty:
                    with tab_sum:
                        st.write("Errors")
                        st.dataframe(err, use_container_width=True)

            # Charts
            with tab_charts:
                cols = st.columns(2)
                if "total" in df.columns:
                    cols[0].subheader("Total Distribution")
                    cols[0].bar_chart(df["total"]) 
                by_subject = {s: df[s].mean() for s in settings.subjects if s in df.columns}
                if by_subject:
                    cols[1].subheader("Per-subject Average")
                    cols[1].bar_chart(by_subject)

            # Downloads
            with tab_dl:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df_for_export.to_excel(writer, index=False, sheet_name="Summary")
                    if not large_batch:
                        for name, dfi in detailed_sheets:
                            safe_name = name[:28]
                            dfi.to_excel(writer, index=False, sheet_name=f"{safe_name}")
                st.download_button(
                    label="Download Excel (Summary{} )".format(" + per-image sheets" if not large_batch else ""),
                    data=buffer.getvalue(),
                    file_name="omr_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                if large_batch:
                    st.caption("Large batch mode active: per-image Excel sheets omitted for performance.")
                csv = df_for_export.to_csv(index=False)
                st.download_button("Download CSV (Summary)", csv, "omr_results.csv", "text/csv")

            if "total" in df.columns:
                st.subheader("Total score distribution")
                st.bar_chart(df["total"]) 
else:
    st.info("Use the uploader above to select one or more images.")
