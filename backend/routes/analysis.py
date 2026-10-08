"""
Health Report Explainer - Analysis Routes
"""
import json
import logging
import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.config import settings
from backend.models.report import AnalyzeRequest, TranslateRequest
from backend.services.ai_service import AIServiceError, ai_service
from backend.services.document_service import document_service
from backend.utils.validators import (
    get_file_type,
    sanitize_filename,
    validate_file_extension,
    validate_file_size,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _normalize_language(language: str) -> str:
    candidate = (language or "english").strip()
    if not candidate:
        return "english"

    key = candidate.lower()
    lookup = {name.lower(): code for code, name in settings.SUPPORTED_LANGUAGES.items()}
    for display_name in settings.SUPPORTED_LANGUAGES.values():
        compact = display_name.lower().replace("(", "").replace(")", "").replace(" ", "")
        lookup[compact] = next(code for code, name in settings.SUPPORTED_LANGUAGES.items() if name == display_name)

    if key in lookup:
        return lookup[key]

    if key in {code.lower() for code in settings.SUPPORTED_LANGUAGES}:
        return key

    raise ValueError(
        f"Unsupported language '{language}'. Please choose one of: "
        + ", ".join(settings.SUPPORTED_LANGUAGES.keys())
    )


@router.post("/analyze")
async def analyze_report(
    request: Request,
    file: UploadFile | None = File(default=None),
    language: str = Form(default="english"),
    extracted_text: str = Form(default=""),
):
    """Analyze a report uploaded as multipart form-data or accept legacy JSON input."""
    filename = "medical_report"
    report_text = extracted_text.strip()

    if file is not None:
        filename = file.filename or filename
        valid, error = validate_file_extension(filename)
        if not valid:
            raise HTTPException(status_code=400, detail=error)

        content = await file.read()
        valid, error = validate_file_size(len(content))
        if not valid:
            raise HTTPException(status_code=400, detail=error)

        safe_name = sanitize_filename(filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        upload_dir = settings.UPLOAD_DIR
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, unique_name)

        try:
            with open(file_path, "wb") as handle:
                handle.write(content)
            report_text = await document_service.extract_text(file_path, safe_name)
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
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    logger.warning("Could not delete temp file: %s", file_path)

    elif not report_text:
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="No report content was provided.")

        report_text = (payload.get("extracted_text") or "").strip()
        filename = payload.get("filename") or filename
        language = payload.get("language") or language

    if not report_text or len(report_text.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="The extracted text is too short to analyze. Please upload a complete medical report.",
        )

    try:
        normalized_language = _normalize_language(language)
        result = await ai_service.analyze_report(
            extracted_text=report_text,
            language=normalized_language,
            report_name=filename,
        )
        return {
            "success": True,
            "language": result["language"],
            "explanation": result["explanation"],
            "report_name": result["report_name"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AIServiceError as exc:
        logger.exception("Analysis failed due to Gemini upstream error: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except Exception as exc:
        logger.exception("Unexpected analysis failure.")
        raise HTTPException(
            status_code=500,
            detail="Unable to analyze the report right now. Please try again.",
        ) from exc


@router.post("/translate")
async def translate_analysis(request: TranslateRequest):
    """Translate or rephrase an existing explanation in a different language."""
    if not request.analysis and not request.extracted_text:
        raise HTTPException(
            status_code=400,
            detail="No analysis or extracted text provided for translation.",
        )

    try:
        translated = await ai_service.translate_analysis(
            analysis=request.analysis,
            target_language=request.target_language,
            extracted_text=request.extracted_text,
            report_name="medical_report",
        )
        return {
            "success": True,
            "language": translated["language"],
            "explanation": translated["explanation"],
            "report_name": translated.get("report_name", "medical_report"),
        }
    except AIServiceError as exc:
        logger.exception("Translation failed due to Gemini upstream error: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected translation failure.")
        raise HTTPException(
            status_code=500,
            detail="Unable to translate the analysis right now. Please try again.",
        ) from exc
