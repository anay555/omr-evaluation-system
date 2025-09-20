from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from PIL import Image
import numpy as np
from app.services.omr import evaluate_image

router = APIRouter(tags=["evaluate"]) 

@router.post("/evaluate")
async def evaluate(sheet_version: str = Form(...), file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        np_img = np.array(image)
        result = evaluate_image(np_img, sheet_version)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or processing error: {e}")
