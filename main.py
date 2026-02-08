from fastapi import FastAPI
from detector import PPEDetector
from risk_engine import RiskEngine
from utils import is_overlap, object_belongs_to_person
import numpy as np

app = FastAPI()

# Load YOLO model
detector = PPEDetector("best.pt")

# Persistent risk tracking per person (temporal)
person_risks = {}
person_risks.clear()

# Class mapping (YOLO class IDs)
CLASS_MAP = {
    0: "helmet",
    2: "vest",
    6: "person",
    7: "no_helmet"
}


@app.post("/analyse")
def analyze_frame(image_path: str):
    detection = detector.detect(image_path)

    # Separate detections
    persons = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "person"]
    helmets = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "helmet"]
    vests = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "vest"]
    no_helmets = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "no_helmet"]

    response = []

    # ---- GLOBAL COUNTERS (WHAT YOU ASKED FOR) ----
    helmet_violation_count = 0
    vest_violation_count = 0

    for idx, person in enumerate(persons):
        person_id = f"person_{idx}"
        pbox = person["bbox"]

        if person_id not in person_risks:
            person_risks[person_id] = RiskEngine()

        engine = person_risks[person_id]

        # ---------------- HELMET CHECK ----------------
        helmet_found = any(
            object_belongs_to_person(pbox, h["bbox"]) for h in helmets
        )

        explicit_no_helmet = any(
            is_overlap(pbox, nh["bbox"]) for nh in no_helmets
        )

        helmet_violation = False
        if not helmet_found or explicit_no_helmet:
            helmet_violation = True
            helmet_violation_count += 1
            engine.add_violation("helmet")

        # ---------------- VEST CHECK ----------------
        vest_found = any(
            is_overlap(pbox, v["bbox"]) for v in vests
        )

        vest_violation = False
        if not vest_found:
            vest_violation = True
            vest_violation_count += 1
            engine.add_violation("harness")

        # ---------------- PERSON RESPONSE ----------------
        # response.append(
        #     {
        #         "person_id": person_id,
        #         "helmet_violation": helmet_violation,
        #         "vest_violation": vest_violation,
        #         # "risk_score": engine.compute_risk_score(),
        #         # "risk_level": engine.escalation_level()
        #     }
        # )

    # ---------------- SITE LEVEL METRICS ----------------
    # site_risk = float(np.mean([p["risk_score"] for p in response])) if response else 0.0

    # ---------------- FINAL JSON OUTPUT ----------------
    return {
        "no_of_person": len(persons),
        "no_of_helmet_violation": helmet_violation_count,
        "no_of_vest_violation": vest_violation_count,
        # "site_risk": site_risk,
        # "persons": response
    }


# Run with:
# uvicorn main:app --reload
