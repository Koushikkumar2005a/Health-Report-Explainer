"""
Health Report Explainer - Configuration Module
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    _raw_api_key: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_KEY: str = _raw_api_key if _raw_api_key and "your_" not in _raw_api_key.lower() else ""
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    GEMINI_FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "temp_uploads")

    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".pdf", ".doc", ".docx"}
    ALLOWED_MIME_TYPES: set = {
        "image/jpeg",
        "image/png",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    SUPPORTED_LANGUAGES: dict = {
        "english": "English",
        "telugu": "Telugu (తెలుగు)",
        "tamil": "Tamil (தமிழ்)",
        "hindi": "Hindi (हिन्दी)",
        "kannada": "Kannada (ಕನ್ನಡ)",
        "malayalam": "Malayalam (മലയാളം)",
        "bengali": "Bengali (বাংলা)",
        "marathi": "Marathi (मराठी)",
        "gujarati": "Gujarati (ગુજરાતી)",
        "punjabi": "Punjabi (ਪੰਜਾਬੀ)",
        "odia": "Odia (ଓଡ଼ିଆ)",
    }


settings = Settings()
