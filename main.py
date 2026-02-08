from fastapi import FastAPI
from detector import PPEDetector
from risk_engine import RiskEngine
from utils import is_overlap, object_belongs_to_person
import numpy as np

app = FastAPI()
detector = None
person_risks = {}

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
def analyze_frame(image_path: str):
    detection = detector.detect(image_path)

    persons = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "person"]
    helmets = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "helmet"]
    vests = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "vest"]
    no_helmets = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "no_helmet"]

    helmet_violation_count = 0
    vest_violation_count = 0

    for idx, person in enumerate(persons):
        person_id = f"person_{idx}"
        pbox = person["bbox"]

        if person_id not in person_risks:
            person_risks[person_id] = RiskEngine()

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
