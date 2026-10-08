"""
Health Report Explainer - Main Application

FastAPI backend server for the Health Report Explainer application.
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.config import settings
from backend.routes import upload, analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Health Report Explainer",
    description="AI-powered medical report explanation tool",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Health Report Explainer",
        "ai_configured": bool(settings.GEMINI_API_KEY),
    }


@app.get("/api/languages")
async def get_languages():
    """Return list of supported languages."""
    return {"languages": settings.SUPPORTED_LANGUAGES}


@app.get("/api/sample-report")
async def get_sample_report():
    """Return a sample fictional medical report for demo purposes."""
    sample = """
CITY GENERAL HOSPITAL & DIAGNOSTIC CENTER
123 Healthcare Avenue, Hyderabad, Telangana - 500001
Phone: +91-40-XXXXXXXX

====================================
COMPLETE BLOOD COUNT (CBC) REPORT
====================================

Patient Name: Demo Patient
Age: 35 Years
Gender: Male
Patient ID: DEMO-2024-001
Referring Doctor: Dr. Example
Sample Collection Date: 15-Sep-2024
Report Date: 16-Sep-2024
Sample Type: Whole Blood (EDTA)

------------------------------------
TEST RESULTS
------------------------------------

TEST NAME                | RESULT    | UNIT        | REFERENCE RANGE     | STATUS
-------------------------|-----------|-------------|---------------------|--------
Hemoglobin (Hb)          | 10.2      | g/dL        | 13.0 - 17.0         | LOW
Total RBC Count          | 4.1       | million/µL  | 4.5 - 5.5           | LOW
Packed Cell Volume (PCV) | 32.5      | %           | 40 - 50             | LOW
MCV                      | 79.3      | fL          | 83 - 101            | LOW
MCH                      | 24.9      | pg          | 27 - 32             | LOW
MCHC                     | 31.4      | g/dL        | 31.5 - 34.5         | LOW
Total WBC Count          | 7,800     | /µL         | 4,000 - 11,000      | NORMAL
Neutrophils              | 62        | %           | 40 - 80             | NORMAL
Lymphocytes              | 30        | %           | 20 - 40             | NORMAL
Eosinophils              | 4         | %           | 1 - 6               | NORMAL
Monocytes                | 3         | %           | 2 - 10              | NORMAL
Basophils                | 1         | %           | 0 - 2               | NORMAL
Platelet Count           | 185,000   | /µL         | 150,000 - 400,000   | NORMAL
RDW                      | 16.8      | %           | 11.5 - 14.5         | HIGH
ESR                      | 28        | mm/hr       | 0 - 15              | HIGH

------------------------------------
IRON STUDIES
------------------------------------

TEST NAME                | RESULT    | UNIT        | REFERENCE RANGE     | STATUS
-------------------------|-----------|-------------|---------------------|--------
Serum Iron               | 42        | µg/dL       | 60 - 170            | LOW
TIBC                     | 420       | µg/dL       | 250 - 370           | HIGH
Ferritin                 | 8         | ng/mL       | 12 - 300            | LOW
Transferrin Saturation   | 10        | %           | 20 - 50             | LOW

------------------------------------
LIVER FUNCTION TESTS
------------------------------------

TEST NAME                | RESULT    | UNIT        | REFERENCE RANGE     | STATUS
-------------------------|-----------|-------------|---------------------|--------
Total Bilirubin          | 0.8       | mg/dL       | 0.1 - 1.2           | NORMAL
Direct Bilirubin         | 0.2       | mg/dL       | 0.0 - 0.3           | NORMAL
SGPT (ALT)               | 22        | U/L         | 7 - 56              | NORMAL
SGOT (AST)               | 19        | U/L         | 10 - 40             | NORMAL
Alkaline Phosphatase     | 78        | U/L         | 44 - 147            | NORMAL
Total Protein            | 7.1       | g/dL        | 6.0 - 8.3           | NORMAL
Albumin                  | 4.2       | g/dL        | 3.5 - 5.5           | NORMAL

------------------------------------
KIDNEY FUNCTION TESTS
------------------------------------

TEST NAME                | RESULT    | UNIT        | REFERENCE RANGE     | STATUS
-------------------------|-----------|-------------|---------------------|--------
Blood Urea              | 32        | mg/dL       | 17 - 43             | NORMAL
Serum Creatinine        | 1.0       | mg/dL       | 0.7 - 1.3           | NORMAL
Uric Acid               | 5.8       | mg/dL       | 3.4 - 7.0           | NORMAL
BUN                      | 15        | mg/dL       | 6 - 20              | NORMAL

------------------------------------
BLOOD SUGAR
------------------------------------

TEST NAME                | RESULT    | UNIT        | REFERENCE RANGE     | STATUS
-------------------------|-----------|-------------|---------------------|--------
Fasting Blood Sugar      | 108       | mg/dL       | 70 - 100            | HIGH
HbA1c                    | 6.1       | %           | 4.0 - 5.6           | HIGH

====================================
NOTE: This is a FICTIONAL sample report created for demonstration purposes.
No real patient data is used. All values are illustrative.

Pathologist: Dr. Sample Pathologist, MD (Pathology)
Lab Technician: Mr. Demo Technician, DMLT
====================================
"""
    return {"success": True, "extracted_text": sample.strip(), "filename": "sample_report.txt", "file_type": "Sample Report"}


# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dir, "assets")), name="assets")
    app.mount("/css", StaticFiles(directory=os.path.join(frontend_dir, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(frontend_dir, "js")), name="js")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/{path:path}")
    async def serve_frontend_pages(path: str):
        # Check if it's a page request
        file_path = os.path.join(frontend_dir, path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        # Default to index.html for SPA routing
        return FileResponse(os.path.join(frontend_dir, "index.html"))


# Create upload directory
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
