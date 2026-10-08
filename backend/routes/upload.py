"""
Health Report Explainer - Upload Routes
"""
import os
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.config import settings
from backend.utils.validators import (
    validate_file_extension,
    validate_file_size,
    sanitize_filename,
    get_file_type,
)
from backend.services.document_service import document_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a medical report file.
    Validates, saves temporarily, extracts text, then cleans up.
    """
    # Validate extension
    valid, error = validate_file_extension(file.filename or "")
    if not valid:
        raise HTTPException(status_code=400, detail=error)

    # Read file content
    content = await file.read()

    # Validate size
    valid, error = validate_file_size(len(content))
    if not valid:
        raise HTTPException(status_code=400, detail=error)

    # Sanitize filename and create temp path
    safe_name = sanitize_filename(file.filename or "upload")
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, unique_name)

    extracted_text = ""
    try:
        # Save temporarily
        with open(file_path, "wb") as f:
            f.write(content)

        # Extract text
        extracted_text = await document_service.extract_text(file_path, safe_name)

    except ValueError as exc:
        logger.exception("Upload extraction failed for %s", safe_name)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Upload processing failed for %s", safe_name)
        raise HTTPException(
            status_code=500,
            detail="Unable to process the uploaded file. Please try again.",
        ) from exc
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                logger.warning(f"Could not delete temp file: {file_path}")

    return {
        "success": True,
        "filename": file.filename,
        "file_type": get_file_type(file.filename or ""),
        "file_size": len(content),
        "extracted_text": extracted_text,
        "message": "Report uploaded and text extracted successfully.",
    }
