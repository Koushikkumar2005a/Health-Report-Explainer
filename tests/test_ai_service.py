import asyncio
import json
import unittest
from unittest.mock import patch

from backend.config import settings
from backend.services.ai_service import AIService, AIServiceError


class FakeResponse:
    def __init__(self, payload):
        self.text = payload if isinstance(payload, str) else json.dumps(payload)


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        response = """### Overall Summary
This report shows low hemoglobin and low ferritin levels, which can be associated with anemia or iron deficiency.

### Important Findings
- Hemoglobin: 10.2 g/dL (reference range 13.0-17.0 g/dL) is low.
- Ferritin: 8 ng/mL (reference range 12-300 ng/mL) is low.

### What This Means
The low hemoglobin and ferritin suggest that the body may not be making enough healthy red blood cells or may not have enough iron stores.

### What to Discuss With Your Doctor
Ask about iron studies, diet, and whether further evaluation is needed.

### Important Notice
This explanation is for informational purposes only and is not a medical diagnosis or a substitute for professional medical advice.
"""
        return FakeResponse(response)


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


class RetryThenFallbackModels:
    def __init__(self, fail_fallback=False):
        self.calls = []
        self.fail_fallback = fail_fallback

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(model)
        if model == "primary-test" or self.fail_fallback:
            raise RuntimeError("503 UNAVAILABLE: high demand")
        return FakeResponse("Fallback explanation")


class RetryThenFallbackClient:
    def __init__(self, fail_fallback=False):
        self.models = RetryThenFallbackModels(fail_fallback)


class AIServiceTests(unittest.TestCase):
    def test_analyze_report_with_no_tests_uses_gemini_only_fallback(self):
        service = AIService()
        service.client = FakeClient()

        report = "Neurosurgery report describes the reason for admission and procedure."
        with patch(
            "backend.services.ai_service.retrieve_report_context",
            return_value={
                "terms": [],
                "tests": [],
                "passages": [],
                "sources": [],
                "retrieval_status": "no_tests_detected",
                "rag_used": False,
            },
        ):
            analysis = asyncio.run(service.analyze_report(report, "telugu"))

        self.assertEqual(analysis["language"], "telugu")
        self.assertEqual(analysis["retrieval_status"], "gemini_only")
        self.assertFalse(analysis["rag_used"])
        self.assertEqual(analysis["retrieved_sources"], [])
        self.assertEqual(analysis["test_explanations"], [])
        self.assertEqual(len(service.client.models.calls), 1)
        call = service.client.models.calls[0]
        self.assertIn(report, call["contents"])
        self.assertIn("Telugu", call["contents"])
        self.assertIn("general explanatory context", call["config"]["system_instruction"])
        self.assertNotIn("response_mime_type", call["config"])

    def test_empty_report_returns_clear_error_without_calling_gemini(self):
        service = AIService()
        service.client = FakeClient()

        with self.assertRaisesRegex(ValueError, "No text could be extracted"):
            asyncio.run(service.analyze_report("  \n ", "english"))

        self.assertEqual(service.client.models.calls, [])

    def test_gemini_only_api_failure_is_propagated(self):
        service = AIService()
        service.client = RetryThenFallbackClient(fail_fallback=True)
        retrieval = {
            "terms": [],
            "tests": [],
            "passages": [],
            "sources": [],
            "retrieval_status": "no_tests_detected",
            "rag_used": False,
        }

        with (
            patch(
                "backend.services.ai_service.retrieve_report_context",
                return_value=retrieval,
            ),
            patch.object(settings, "GEMINI_MODEL", "primary-test"),
            patch.object(settings, "GEMINI_FALLBACK_MODEL", ""),
            patch("backend.services.ai_service.time.sleep"),
        ):
            with self.assertRaises(AIServiceError):
                asyncio.run(
                    service.analyze_report(
                        "A sufficiently long neurosurgery report text for analysis.",
                        "english",
                    )
                )

    def test_retries_primary_then_uses_fallback(self):
        service = AIService()
        service.client = RetryThenFallbackClient()

        with (
            patch.object(settings, "GEMINI_MODEL", "primary-test"),
            patch.object(settings, "GEMINI_FALLBACK_MODEL", "fallback-test"),
            patch("backend.services.ai_service.time.sleep"),
        ):
            response = service._generate("test prompt")

        self.assertEqual(response.text, "Fallback explanation")
        self.assertEqual(
            service.client.models.calls,
            ["primary-test"] * 3 + ["fallback-test"],
        )

    def test_both_models_unavailable_returns_clean_503(self):
        service = AIService()
        service.client = RetryThenFallbackClient(fail_fallback=True)

        with (
            patch.object(settings, "GEMINI_MODEL", "primary-test"),
            patch.object(settings, "GEMINI_FALLBACK_MODEL", "fallback-test"),
            patch("backend.services.ai_service.time.sleep"),
        ):
            with self.assertRaises(AIServiceError) as raised:
                service._generate("test prompt")

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            raised.exception.detail,
            "Gemini is temporarily experiencing high demand. Please try again in a few minutes.",
        )
        self.assertEqual(
            service.client.models.calls,
            ["primary-test"] * 3 + ["fallback-test"] * 3,
        )


if __name__ == "__main__":
    unittest.main()
