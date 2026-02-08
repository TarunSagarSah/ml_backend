from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="Safety Detection API")

# Allow frontend (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Load YOLO model ONCE
# --------------------------
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

# --------------------------
# Run YOLO on image
# --------------------------
def run_model_on_image(image: Image.Image):
    results = model(image, conf=0.6)  # increase confidence

    people = 0
    helmet_violations = 0
    vest_violations = 0

    for r in results:
        if r.boxes is None:
            continue

        classes = r.boxes.cls.cpu().tolist()

        for cls in classes:
            cls = int(cls)

            # ⚠️ UPDATE THESE IDS BASED ON YOUR FRIEND'S MODEL
            if cls == 6:
                people += 1
            elif cls == 7:
                helmet_violations += 1
            elif cls == 2:
                vest_violations += 1

    return {
        "number_of_people": people,
        "helmet_violations": helmet_violations,
        "vest_violations": vest_violations
    }

# --------------------------
# Prediction endpoint
# --------------------------
@app.post("/predict")
async def predict(frame: UploadFile = File(...)):
    try:
        image_bytes = await frame.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = run_model_on_image(image)
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
