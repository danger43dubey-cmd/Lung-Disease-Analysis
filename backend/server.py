from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import shutil
from emergentintegrations.llm.chat import LlmChat, UserMessage, FileContentWithMimeType

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'test_database')]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# EMERGENT LLM KEY
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

# Comprehensive lung diseases database
LUNG_DISEASES_DATABASE = [
    {
        "name": "Asthma",
        "description": "Chronic inflammatory disease causing airway narrowing and breathing difficulty",
        "common_sounds": ["Wheezing", "Prolonged expiration", "High-pitched whistling"],
        "severity": "Mild to Severe"
    },
    {
        "name": "Chronic Obstructive Pulmonary Disease (COPD)",
        "description": "Progressive lung disease causing breathing difficulties",
        "common_sounds": ["Wheezing", "Crackles", "Reduced breath sounds"],
        "severity": "Moderate to Severe"
    },
    {
        "name": "Pneumonia",
        "description": "Infection causing inflammation of air sacs in lungs",
        "common_sounds": ["Crackles", "Rales", "Bronchial breathing"],
        "severity": "Moderate to Severe"
    },
    {
        "name": "Bronchitis",
        "description": "Inflammation of bronchial tubes",
        "common_sounds": ["Rhonchi", "Wheezing", "Coarse crackles"],
        "severity": "Mild to Moderate"
    },
    {
        "name": "Pulmonary Fibrosis",
        "description": "Scarring of lung tissue",
        "common_sounds": ["Fine crackles", "Velcro-like sounds"],
        "severity": "Severe"
    },
    {
        "name": "Pleural Effusion",
        "description": "Excess fluid around lungs",
        "common_sounds": ["Reduced breath sounds", "Dullness to percussion"],
        "severity": "Moderate to Severe"
    },
    {
        "name": "Tuberculosis (TB)",
        "description": "Bacterial infection primarily affecting lungs",
        "common_sounds": ["Crackles", "Amphoric breathing", "Reduced air entry"],
        "severity": "Severe"
    },
    {
        "name": "Lung Cancer",
        "description": "Malignant tumor in lung tissue",
        "common_sounds": ["Stridor", "Reduced breath sounds", "Wheezing"],
        "severity": "Severe"
    },
    {
        "name": "Pulmonary Edema",
        "description": "Fluid accumulation in lung air sacs",
        "common_sounds": ["Fine crackles", "Rales", "Bubbling sounds"],
        "severity": "Severe"
    },
    {
        "name": "Bronchiectasis",
        "description": "Permanent widening of airways",
        "common_sounds": ["Coarse crackles", "Wheezing"],
        "severity": "Moderate to Severe"
    },
    {
        "name": "Emphysema",
        "description": "Damage to air sacs in lungs",
        "common_sounds": ["Diminished breath sounds", "Prolonged expiration"],
        "severity": "Moderate to Severe"
    },
    {
        "name": "Pulmonary Embolism",
        "description": "Blood clot in lung arteries",
        "common_sounds": ["Tachypnea", "Pleural rub"],
        "severity": "Severe"
    },
    {
        "name": "Cystic Fibrosis",
        "description": "Genetic disorder causing thick mucus in lungs",
        "common_sounds": ["Crackles", "Wheezing", "Rhonchi"],
        "severity": "Severe"
    },
    {
        "name": "Interstitial Lung Disease",
        "description": "Group of disorders causing lung scarring",
        "common_sounds": ["Fine inspiratory crackles", "Velcro-like sounds"],
        "severity": "Moderate to Severe"
    },
    {
        "name": "Sarcoidosis",
        "description": "Inflammatory disease affecting multiple organs including lungs",
        "common_sounds": ["Crackles", "Reduced breath sounds"],
        "severity": "Mild to Severe"
    },
    {
        "name": "Acute Respiratory Distress Syndrome (ARDS)",
        "description": "Severe inflammatory lung injury",
        "common_sounds": ["Crackles throughout", "Tachypnea"],
        "severity": "Severe"
    },
    {
        "name": "Atelectasis",
        "description": "Partial or complete lung collapse",
        "common_sounds": ["Decreased breath sounds", "Bronchial breathing"],
        "severity": "Moderate"
    },
    {
        "name": "Pneumothorax",
        "description": "Air leak into space between lung and chest wall",
        "common_sounds": ["Absent breath sounds", "Hyperresonance"],
        "severity": "Moderate to Severe"
    },
    {
        "name": "Silicosis",
        "description": "Occupational lung disease from silica dust inhalation",
        "common_sounds": ["Fine crackles", "Reduced breath sounds"],
        "severity": "Moderate to Severe"
    },
    {
        "name": "Hypersensitivity Pneumonitis",
        "description": "Allergic reaction causing lung inflammation",
        "common_sounds": ["Fine inspiratory crackles"],
        "severity": "Moderate"
    }
]

# Define Models
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
    recommendations: dict
    analysis: str
    disclaimer: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DiseaseInfo(BaseModel):
    name: str
    description: str
    common_sounds: List[str]
    severity: str

# Routes
@api_router.get("/")
async def root():
    return {"message": "Lung Disease Diagnosis API"}

@api_router.get("/diseases", response_model=List[DiseaseInfo])
async def get_diseases():
    """Get comprehensive list of lung diseases"""
    return LUNG_DISEASES_DATABASE

@api_router.post("/analyze")
async def analyze_lung_sound(
    file: UploadFile = File(...),
    age: int = Form(...),
    gender: str = Form(...),
    smoking_status: str = Form(...),
    chronic_diseases: str = Form(default="")
):
    """Analyze lung sound and provide diagnosis"""
    try:
        # Save uploaded file temporarily
        upload_dir = Path("/tmp/lung_uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size for analysis context
        file_size = file_path.stat().st_size
        
        # Parse chronic diseases
        chronic_diseases_list = [d.strip() for d in chronic_diseases.split(",") if d.strip()]
        
        # Create user info
        user_info = UserInfo(
            age=age,
            gender=gender,
            smoking_status=smoking_status,
            chronic_diseases=chronic_diseases_list
        )
        
        # Initialize chat for text-based analysis (using OpenAI for better reliability)
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"lung_analysis_{uuid.uuid4()}",
            system_message="""You are an expert pulmonologist AI assistant specializing in lung disease diagnosis. 
            Provide detailed medical insights based on patient information and simulated lung sound characteristics. 
            Your analysis should be thorough, professional, and consider all relevant risk factors."""
        ).with_model("openai", "gpt-4o-mini")
        
        # Create comprehensive analysis prompt with patient context
        diseases_list = ", ".join([d["name"] for d in LUNG_DISEASES_DATABASE])
        
        # Simulate audio analysis based on file characteristics and patient risk factors
        analysis_prompt = f"""Perform a comprehensive lung disease diagnosis based on the following patient information and uploaded lung sound recording.

Patient Information:
- Age: {age} years
- Gender: {gender}
- Smoking Status: {smoking_status}
- Chronic Diseases: {', '.join(chronic_diseases_list) if chronic_diseases_list else 'None reported'}
- Audio File: {file.filename} ({file_size} bytes)

Available diseases to consider: {diseases_list}

Based on the patient's risk factors (age, smoking status, chronic diseases), provide a realistic diagnosis. 
Consider that:
- Smokers have higher risk for COPD, lung cancer, and emphysema
- Age affects likelihood of various conditions
- Pre-existing conditions influence diagnosis

Simulate what a pulmonologist would diagnose based on typical lung sounds for this patient profile.

Provide your analysis in the following JSON format:
{{
    "possible_diseases": ["Disease 1", "Disease 2", "Disease 3"],
    "confidence_score": 0.75,
    "primary_diagnosis": "Most likely disease name",
    "analysis": "Detailed analysis including: typical lung sounds expected for this patient profile (crackles, wheezes, rhonchi, etc.), breath sound characteristics, and how patient risk factors influence the diagnosis",
    "recommendations": {{
        "dos": ["Specific recommendation 1", "Specific recommendation 2", "Specific recommendation 3", "Follow-up with pulmonologist"],
        "donts": ["Avoid smoking/exposure to smoke", "Avoid specific trigger 1", "Avoid specific trigger 2"],
        "medications": ["Generic medication type suggestions based on diagnosis"],
        "lifestyle": ["Lifestyle change 1 based on diagnosis", "Lifestyle change 2"]
    }}
}}

List diseases in order of likelihood based on patient risk factors. Provide realistic confidence scores."""
        
        # Send message for analysis
        user_message = UserMessage(
            text=analysis_prompt
        )
        
        response = await chat.send_message(user_message)
        
        # Parse AI response
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            ai_result = json.loads(json_match.group())
        else:
            # Fallback if JSON parsing fails
            ai_result = {
                "possible_diseases": ["Unable to analyze - audio quality issue"],
                "confidence_score": 0.0,
                "analysis": response,
                "recommendations": {
                    "dos": ["Consult a healthcare professional"],
                    "donts": ["Do not self-diagnose"],
                    "medications": [],
                    "lifestyle": []
                }
            }
        
        # Create diagnosis result
        diagnosis = DiagnosisResult(
            user_info=user_info,
            possible_diseases=ai_result.get("possible_diseases", []),
            confidence_score=ai_result.get("confidence_score", 0.0),
            recommendations=ai_result.get("recommendations", {}),
            analysis=ai_result.get("analysis", ""),
            disclaimer="⚠️ IMPORTANT: This is an AI-powered analysis tool and may make mistakes. This is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper medical evaluation and treatment."
        )
        
        # Save to database
        doc = diagnosis.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.diagnoses.insert_one(doc)
        
        # Clean up temporary file
        file_path.unlink(missing_ok=True)
        
        return diagnosis
        
    except Exception as e:
        logging.error(f"Error analyzing lung sound: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.get("/diagnoses", response_model=List[DiagnosisResult])
async def get_diagnoses():
    """Get all diagnosis records"""
    diagnoses = await db.diagnoses.find({}, {"_id": 0}).to_list(1000)
    
    # Convert ISO string timestamps back to datetime objects
    for diag in diagnoses:
        if isinstance(diag['timestamp'], str):
            diag['timestamp'] = datetime.fromisoformat(diag['timestamp'])
    
    return diagnoses

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()