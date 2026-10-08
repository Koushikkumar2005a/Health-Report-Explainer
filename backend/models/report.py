"""
Health Report Explainer - Data Models
"""
from pydantic import BaseModel
from typing import Optional, List


class Finding(BaseModel):
    """Individual test finding from a medical report."""
    test_name: str = ""
    value: str = ""
    unit: str = ""
    reference_range: str = ""
    status: str = ""  # Normal, High, Low, Critical
    simple_explanation: str = ""
    why_it_matters: str = ""


class PatientInfo(BaseModel):
    """Patient information extracted from report."""
    name: str = "Not available in the report"
    age: str = "Not available in the report"
    gender: str = "Not available in the report"
    report_date: str = "Not available in the report"
    hospital_name: str = "Not available in the report"


class MedicalTerm(BaseModel):
    """Medical term with explanation."""
    term: str = ""
    explanation: str = ""


class ReportAnalysis(BaseModel):
    """Complete AI analysis of a medical report."""
    summary: str = ""
    patient_information: PatientInfo = PatientInfo()
    overall_attention: str = "Normal"  # Normal, Attention, Discuss with Doctor, Urgent Attention
    findings: List[Finding] = []
    normal_results: List[Finding] = []
    abnormal_results: List[Finding] = []
    medical_terms: List[MedicalTerm] = []
    possible_interpretations: List[str] = []
    doctor_questions: List[str] = []
    urgent_notice: str = ""
    disclaimer: str = (
        "This tool provides an AI-generated explanation of the information contained in your report. "
        "It is for educational and informational purposes only and is not a medical diagnosis or a "
        "substitute for professional medical advice. Always consult a qualified healthcare professional "
        "for diagnosis and treatment."
    )
    language: str = "english"


class AnalyzeRequest(BaseModel):
    """Request model for report analysis."""
    extracted_text: str
    language: str = "english"
    filename: str = ""


class TranslateRequest(BaseModel):
    """Request model for translating an existing analysis."""
    analysis: dict
    target_language: str
    extracted_text: str = ""


class UploadResponse(BaseModel):
    """Response model for file upload."""
    success: bool
    filename: str = ""
    file_type: str = ""
    file_size: int = 0
    extracted_text: str = ""
    message: str = ""
    error: str = ""
