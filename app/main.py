from fastapi import FastAPI
from app.routers.evaluate import router as evaluate_router
from app.routers.results import router as results_router

app = FastAPI(title="OMR Evaluation API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(evaluate_router, prefix="/api")
app.include_router(results_router, prefix="/api")
