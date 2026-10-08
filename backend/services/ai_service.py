"""
Health Report Explainer - AI Service

Abstracted AI service for medical report analysis.
Currently uses the Google Gemini API. Can be swapped to another provider
by implementing the same interface.
"""
import logging
import time
from google import genai
from backend.config import settings

logger = logging.getLogger(__name__)

_GEMINI_RETRY_ATTEMPTS = 3
_TEMPORARY_ERROR_MARKERS = (
    "503",
    "unavailable",
    "temporarily unavailable",
    "high demand",
    "rate limit",
    "429",
    "resource exhausted",
    "too many requests",
)


class AIServiceError(RuntimeError):
    """Raised when the upstream Gemini API fails."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.detail = message
        self.status_code = status_code


def _is_retryable_gemini_error(exc: Exception) -> bool:
    """Determine whether the failure is likely temporary and safe to retry."""
    if exc is None:
        return False

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {429, 503}:
        return True

    error_code = getattr(exc, "code", None)
    if isinstance(error_code, int) and error_code in {429, 503}:
        return True

    message = str(exc).lower()
    return any(marker in message for marker in _TEMPORARY_ERROR_MARKERS)


def _clean_gemini_failure_message() -> str:
    return "Gemini is temporarily experiencing high demand. Please try again in a few minutes."


def _model_sequence() -> list[str]:
    models = []
    primary = (settings.GEMINI_MODEL or "gemini-3.8-flash").strip()
    fallback = (settings.GEMINI_FALLBACK_MODEL or "").strip()

    if primary:
        models.append(primary)
    if fallback and fallback.lower() != primary.lower():
        models.append(fallback)
    return models


def _normalize_language(language: str) -> str:
    """Normalize the user-selected language to a supported key."""
    candidate = (language or "english").strip()
    if not candidate:
        return "english"

    key = candidate.lower()
    lookup = {name.lower(): code for code, name in settings.SUPPORTED_LANGUAGES.items()}
    for display_name in settings.SUPPORTED_LANGUAGES.values():
        compact = display_name.lower().replace("(", "").replace(")", "").replace(" ", "")
        lookup[compact] = next(code for code, name in settings.SUPPORTED_LANGUAGES.items() if name == display_name)

    return lookup.get(key, key if key in {code.lower() for code in settings.SUPPORTED_LANGUAGES} else "english")


def _display_language(language: str) -> str:
    key = _normalize_language(language)
    display = settings.SUPPORTED_LANGUAGES.get(key, "English")
    return display.split(" (")[0]


def _build_analysis_prompt(extracted_text: str, language: str) -> str:
    """Build the medical report explanation prompt with explicit language and safety rules."""
    normalized_language = _normalize_language(language)
    display_language = _display_language(normalized_language)

    prompt = f"""You are a medical report explainer AI for educational use.

The user has uploaded a medical report.
Analyze the report carefully and explain it in simple, easy-to-understand language.

IMPORTANT INSTRUCTIONS:
- The selected response language is: {display_language}
- Respond entirely in {display_language}. Do not switch to English.
- Preserve all numerical values, units, reference ranges, test names, dates, and factual medical details exactly as reported.
- Explain medical terms simply but do not change clinical facts.
- Do not diagnose the patient.
- Do not prescribe medicines, recommend dosage changes, or tell the user to stop medicines.
- Do not invent missing tests, values, symptoms, or patient information.
- If something is not available in the report, say "Not available in the report".
- If the findings are potentially serious or urgent, clearly suggest seeking appropriate medical attention.
- This explanation is educational and informational. It is not a medical diagnosis.

Use the following structure in your response with headings:
### Overall Summary
### Important Findings
### What This Means
### What to Discuss With Your Doctor
### Important Notice

The explanation should read like a clear patient-friendly report summary, not a diagnosis.
It should mention when findings may need review by a doctor and include a short safety warning.

Medical report content:
\"\"\"
{extracted_text}
\"\"\"

Return plain readable text only. Do not use JSON, markdown tables, or code fences.
End with a disclaimer that says this explanation is for informational purposes only and is not a medical diagnosis or a substitute for professional medical advice.
"""
    return prompt


class AIService:
    """Abstracted AI service for medical report analysis."""

    def __init__(self):
        self.client = None
        if settings.GEMINI_API_KEY:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def _generate_with_model(self, model_name: str, prompt: str):
        """Attempt a Gemini request against one model, retrying only temporary service failures."""
        if not self.client:
            raise ValueError(
                "AI service is not configured. Please set GEMINI_API_KEY in your .env file."
            )

        last_error = None
        for attempt in range(1, _GEMINI_RETRY_ATTEMPTS + 1):
            try:
                logger.info("Trying Gemini model: %s (attempt %s/%s)", model_name, attempt, _GEMINI_RETRY_ATTEMPTS)
                return self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.2,
                        "max_output_tokens": 4096,
                    },
                )
            except Exception as exc:  # pragma: no cover - exercised via upstream API logging
                last_error = exc
                logger.exception("Gemini API request failed for model=%s (attempt %s/%s)", model_name, attempt, _GEMINI_RETRY_ATTEMPTS)
                if not _is_retryable_gemini_error(exc):
                    raise AIServiceError(f"Gemini API request failed: {exc}") from exc

                if attempt < _GEMINI_RETRY_ATTEMPTS:
                    wait_seconds = 2 ** (attempt - 1)
                    logger.warning("Primary model unavailable, retrying in %s seconds...", wait_seconds)
                    time.sleep(wait_seconds)
                    continue

        if last_error is not None:
            raise AIServiceError(_clean_gemini_failure_message(), status_code=503) from last_error
        raise AIServiceError(_clean_gemini_failure_message(), status_code=503)

    def _generate(self, prompt: str):
        """Run a Gemini generation request against the primary model and fallback model if needed."""
        models = _model_sequence()
        if not models:
            raise ValueError("No Gemini model is configured.")

        last_error = None
        for index, model_name in enumerate(models):
            try:
                if index > 0:
                    logger.warning("Trying fallback Gemini model: %s", model_name)
                return self._generate_with_model(model_name, prompt)
            except AIServiceError as exc:
                last_error = exc
                if not _is_retryable_gemini_error(exc):
                    raise
                if index < len(models) - 1:
                    continue
                raise AIServiceError(_clean_gemini_failure_message(), status_code=503) from exc
            except ValueError:
                raise
            except Exception as exc:
                last_error = AIServiceError(f"Gemini API request failed: {exc}")
                raise last_error

        if last_error is not None:
            raise AIServiceError(_clean_gemini_failure_message(), status_code=503) from last_error
        raise AIServiceError(_clean_gemini_failure_message(), status_code=503)

    async def analyze_report(
        self,
        extracted_text: str,
        language: str = "english",
        report_name: str = "medical_report",
    ) -> dict:
        """Analyze extracted medical report text and return a human-readable explanation."""
        if not extracted_text or len(extracted_text.strip()) < 20:
            raise ValueError(
                "The extracted text is too short to analyze. "
                "Please upload a complete medical report."
            )

        normalized_language = _normalize_language(language)
        prompt = _build_analysis_prompt(extracted_text, normalized_language)

        try:
            response = self._generate(prompt)
            response_text = response.text.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines)

            if not response_text:
                raise AIServiceError("Gemini returned an empty response.", status_code=502)

            return {
                "success": True,
                "language": normalized_language,
                "explanation": response_text.strip(),
                "report_name": report_name or "medical_report",
            }

        except AIServiceError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error during AI report analysis.")
            raise AIServiceError(f"Unable to analyze the report right now: {exc}") from exc

    async def translate_analysis(
        self,
        analysis: dict,
        target_language: str,
        extracted_text: str = "",
        report_name: str = "medical_report",
    ) -> dict:
        """Translate or rephrase an existing explanation into the target language."""
        normalized_language = _normalize_language(target_language)

        if extracted_text:
            return await self.analyze_report(extracted_text, normalized_language, report_name)

        text_to_translate = analysis.get("explanation") if isinstance(analysis, dict) else str(analysis)
        if not text_to_translate:
            raise ValueError("No explanation available to translate.")

        prompt = (
            f"Translate the following medical report explanation into { _display_language(normalized_language) }. "
            "Keep all medical facts and numbers exactly as they are. "
            "Do not diagnose the patient. "
            "Return only the translated explanation in plain readable text with headings.\n\n"
            f"{text_to_translate}"
        )

        try:
            response = self._generate(prompt)
            response_text = response.text.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines)

            return {
                "success": True,
                "language": normalized_language,
                "explanation": response_text.strip(),
                "report_name": report_name or "medical_report",
            }
        except AIServiceError:
            raise
        except Exception as exc:
            logger.exception("Translation failed for target_language=%s", normalized_language)
            raise AIServiceError(
                f"Unable to translate the analysis right now: {exc}",
                status_code=502,
            ) from exc


# Singleton instance
ai_service = AIService()
