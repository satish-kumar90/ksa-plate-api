from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from extractor import KSAPlateExtractor

import os

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
PLATE_MODEL_PATH = os.path.join(MODEL_DIR, "lp_ksa_plate_detection.pt")
OCR_MODEL_PATH = os.path.join(MODEL_DIR, "lp_ksa_ocr.pt")




# Automatically use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
plate_model = YOLO(PLATE_MODEL_PATH)
ocr_extractor = KSAPlateExtractor(OCR_MODEL_PATH, conf=0.25, device=device)
# Load models once
#plate_model = YOLO(PLATE_MODEL_PATH)
#ocr_extractor = KSAPlateExtractor(OCR_MODEL_PATH, conf=0.25, device="cpu")

app = FastAPI()

from db_utils import insert_result
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/extract")
async def extract_plate(file: UploadFile = File(...)):
    np_arr = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return JSONResponse(content={"error": "Unable to decode image."}, status_code=400)

    results = ocr_extractor.detect_and_recognize_plates(img, plate_model)
    if not results:
        return JSONResponse(content={"error": "No license plates detected."}, status_code=422)
    
    # Get the first result (you can modify this if you want to handle multiple plates)
    result = results[0]
    area = result.get('area_code', '')
    number = result.get('license_number', '')
    final_number = result.get('plate_number', '')

    # Format the final number string
    final_number_str = f"{area} {number}"

    # Save to DB
    insert_result(
        image_name=file.filename,
        final_number=final_number_str.strip(),
        area=area,
        number=number,
        others=None,  # Not used in current implementation
        angle=0.0     # Default angle value
    )

    return {
        "final_number": final_number_str.strip(),
        "area": area,
        "number": number,

    }



#uvicorn app:app --reload
