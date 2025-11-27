from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict
from typing import List
from pathlib import Path
from datetime import datetime, timezone
from openai import OpenAI
import shutil
import logging
import os
import uuid
import re
import json

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# AI Client (ChatGPT 5.1)
client_ai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# MongoDB
mongo_url = os.environ.get("MONGO_URL")
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get("DB_NAME", "lung_database")]

# FastAPI
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Models
# ------------------------------

class UserInfo(BaseModel):
    age: int
    gender: str
    smoking_status: str
    chronic_diseases: List[str] = []


class DiagnosisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_info: UserInfo
    possible_diseases: List[str]
    confidence_score: float
    analysis: str
    recommendations: dict
    disclaimer: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ------------------------------
# Lung diseases (for AI context)
# ------------------------------
LUNG_DISEASES = [
    "Asthma", "COPD", "Pneumonia", "Bronchitis", "Pulmonary Fibrosis",
    "Pleural Effusion", "Tuberculosis", "Lung Cancer", "Pulmonary Edema",
    "Bronchiectasis", "Emphysema", "Pulmonary Embolism", "Cystic Fibrosis",
    "Interstitial Lung Disease", "Sarcoidosis", "ARDS", "Atelectasis",
    "Pneumothorax", "Silicosis", "Hypersensitivity Pneumonitis"
]


# ------------------------------
# Routes
# ------------------------------

@api_router.get("/")
async def root():
    return {"message": "Lung Disease Diagnosis API (ChatGPT 5.1 version)"}


@api_router.post("/analyze")
async def analyze_lung_sound(
    file: UploadFile = File(...),
    age: int = Form(...),
    gender: str = Form(...),
    smoking_status: str = Form(...),
    chronic_diseases: str = Form(default="")
):
    """
    Analyze lung sound using ChatGPT 5.1 + patient info.
    """
    try:
        # Save uploaded audio temporarily
        upload_dir = Path("/tmp/lung_uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = file_path.stat().st_size

        # Parse chronic diseases
        chronic_list = [d.strip() for d in chronic_diseases.split(",") if d.strip()]

        # Build user profile object
        user_info = UserInfo(
            age=age,
            gender=gender,
            smoking_status=smoking_status,
            chronic_diseases=chronic_list
        )

        # AI prompt for medical diagnosis
        prompt = f"""
You are a senior pulmonologist AI analyzing a patient's lung sound audio file.

PATIENT INFO:
- Age: {age}
- Gender: {gender}
- Smoking Status: {smoking_status}
- Chronic Diseases: {', '.join(chronic_list) if chronic_list else "None"}
- Audio File Size: {file_size} bytes

DISEASES TO CONSIDER:
{", ".join(LUNG_DISEASES)}

Analyze for:
- wheezing
- crackles (coarse/fine)
- stridor
- diminished breath sounds
- rhonchi
- bronchial breathing
- fluid or mucus signs

Return ONLY valid JSON in this format:

{{
  "possible_diseases": ["Asthma", "COPD"],
  "confidence_score": 0.82,
  "analysis": "Detailed medical explanation...",
  "recommendations": {{
    "dos": ["..."],
    "donts": ["..."],
    "medications": ["..."],
    "lifestyle": ["..."]
  }}
}}
"""

        # Call ChatGPT-5.1
        response = client_ai.responses.create(
            model="gpt-5.1",
            input=[
                {"role": "system", "content": "You are an expert pulmonologist. Always output pure JSON."},
                {"role": "user", "content": prompt}
            ],
        )

        raw_output = response.output_text

        # Extract JSON from AI output
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            ai_data = json.loads(match.group())
        else:
            ai_data = {
                "possible_diseases": ["Unclear"],
                "confidence_score": 0.0,
                "analysis": raw_output,
                "recommendations": {"dos": [], "donts": [], "medications": [], "lifestyle": []},
            }

        # Build final diagnosis object
        diagnosis = DiagnosisResult(
            user_info=user_info,
            possible_diseases=ai_data.get("possible_diseases", []),
            confidence_score=ai_data.get("confidence_score", 0.0),
            analysis=ai_data.get("analysis", ""),
            recommendations=ai_data.get("recommendations", {}),
            disclaimer="⚠️ This is an AI tool. It may be incorrect. Always consult a real doctor."
        )

        # Save to DB
        doc = diagnosis.model_dump()
        doc["timestamp"] = doc["timestamp"].isoformat()
        await db.diagnoses.insert_one(doc)

        # Delete audio
        file_path.unlink(missing_ok=True)

        return diagnosis

    except Exception as e:
        logging.error(f"Error analyzing lung sound: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/diagnoses")
async def get_diagnoses():
    """Fetch all stored diagnoses"""
    data = await db.diagnoses.find({}, {"_id": 0}).to_list(500)
    for d in data:
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
    return data


# Register API Router
app.include_router(api_router)


# MongoDB cleanup
@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
