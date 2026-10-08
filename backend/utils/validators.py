"""
Health Report Explainer - File Validators
"""
import os
from backend.config import settings


def validate_file_extension(filename: str) -> tuple[bool, str]:
    """Validate file has an allowed extension."""
    if not filename:
        return False, "No filename provided."

    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported file type '{ext}'. "
            f"Please upload JPG, JPEG, PNG, PDF, DOC, or DOCX files."
        )
    return True, ""


def validate_file_size(file_size: int) -> tuple[bool, str]:
    """Validate file size is within limits."""
    if file_size == 0:
        return False, "The uploaded file is empty. Please upload a valid report."

    if file_size > settings.MAX_FILE_SIZE_BYTES:
        return False, (
            f"File is too large ({file_size / (1024*1024):.1f} MB). "
            f"Maximum allowed size is {settings.MAX_FILE_SIZE_MB} MB."
        )
    return True, ""


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and other issues."""
    # Remove path components
    filename = os.path.basename(filename)
    # Remove potentially dangerous characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    sanitized = "".join(c if c in safe_chars else "_" for c in filename)
    # Ensure it has an extension
    if "." not in sanitized:
        sanitized += ".unknown"
    return sanitized


def get_file_type(filename: str) -> str:
    """Get human-readable file type from filename."""
    ext = os.path.splitext(filename)[1].lower()
    type_map = {
        ".jpg": "JPEG Image",
        ".jpeg": "JPEG Image",
        ".png": "PNG Image",
        ".pdf": "PDF Document",
        ".doc": "Word Document",
        ".docx": "Word Document",
    }
    return type_map.get(ext, "Unknown")
