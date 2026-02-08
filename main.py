from fastapi import FastAPI, File, UploadFile, HTTPException
from detector import PPEDetector
from utils import is_overlap, object_belongs_to_person
import os

app = FastAPI()
detector = None

CLASS_MAP = {
    0: "helmet",
    2: "vest",
    6: "person",
    7: "no_helmet"
}

@app.on_event("startup")
def load_model():
    global detector
    detector = PPEDetector("best.pt")

@app.post("/analyse")
async def analyze_frame(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        temp_path = f"/tmp/{file.filename}"

        with open(temp_path, "wb") as f:
            f.write(contents)

        detection = detector.detect(temp_path)
        os.remove(temp_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    persons = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "person"]
    helmets = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "helmet"]
    vests = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "vest"]
    no_helmets = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "no_helmet"]

    helmet_violation_count = 0
    vest_violation_count = 0

    for person in persons:
        pbox = person["bbox"]

        if not any(object_belongs_to_person(pbox, h["bbox"]) for h in helmets) \
           or any(is_overlap(pbox, nh["bbox"]) for nh in no_helmets):
            helmet_violation_count += 1

        if not any(is_overlap(pbox, v["bbox"]) for v in vests):
            vest_violation_count += 1

    return {
        "no_of_person": len(persons),
        "no_of_helmet_violation": helmet_violation_count,
        "no_of_vest_violation": vest_violation_count
    }
@app.get("/")
def root():
    return {"status": "Backend is running!"}
