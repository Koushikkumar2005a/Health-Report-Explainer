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
    def __init__(self, response=None):
        self.calls = []
        self.response = response or (
            "The extracted text does not contain enough readable medical findings "
            "for a reliable interpretation."
        )

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return FakeResponse(self.response)


class FakeClient:
    def __init__(self, response=None):
        self.models = FakeModels(response)


class RetryThenFallbackModels:
    def __init__(
        self,
        fail_fallback=False,
        primary_error="503 UNAVAILABLE: high demand",
        fallback_error="503 UNAVAILABLE: high demand",
    ):
        self.calls = []
        self.fail_fallback = fail_fallback
        self.primary_error = primary_error
        self.fallback_error = fallback_error

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(model)
        if model == "primary-test":
            if isinstance(self.primary_error, Exception):
                raise self.primary_error
            raise RuntimeError(self.primary_error)
        if self.fail_fallback:
            raise RuntimeError(self.fallback_error)
        return FakeResponse("Fallback explanation")


class RetryThenFallbackClient:
    def __init__(self, **kwargs):
        self.models = RetryThenFallbackModels(**kwargs)


class QuotaLimitModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(model)
        raise RuntimeError("429 RESOURCE_EXHAUSTED: daily quota exceeded")


class QuotaLimitClient:
    def __init__(self):
        self.models = QuotaLimitModels()


class AIServiceTests(unittest.TestCase):
    def test_analyze_report_with_no_tests_uses_gemini_only_fallback(self):
        service = AIService()
        mock_explanation = (
            "విశ్వసనీయ వైద్య వివరణకు సరిపడే నిర్దిష్ట వైద్య ఫలితాలు "
            "ఈ OCR పాఠ్యంలో లేవు."
        )
        service.client = FakeClient(mock_explanation)

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
        self.assertEqual(analysis["explanation"], mock_explanation)
        self.assertEqual(len(service.client.models.calls), 1)
        call = service.client.models.calls[0]
        self.assertIn(report, call["contents"])
        self.assertIn("Telugu", call["contents"])
        self.assertIn("Report Facts", call["contents"])
        self.assertIn("only one short, honest", call["contents"])
        self.assertIn("generic medical or textbook information", call["contents"])
        self.assertIn("specific field or value", call["contents"])
        self.assertIn("general explanatory context", call["config"]["system_instruction"])
        self.assertIn("only source for report-specific facts", call["config"]["system_instruction"])
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
        ):
            with self.assertRaises(AIServiceError) as raised:
                asyncio.run(
                    service.analyze_report(
                        "A sufficiently long neurosurgery report text for analysis.",
                        "english",
                    )
                )
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(service.client.models.calls, ["primary-test"])

    def test_503_uses_fallback_once_and_returns_success(self):
        service = AIService()
        service.client = RetryThenFallbackClient()

        with (
            patch.object(settings, "GEMINI_MODEL", "primary-test"),
            patch.object(settings, "GEMINI_FALLBACK_MODEL", "fallback-test"),
        ):
            response = service._generate("test prompt")

        self.assertEqual(response.text, "Fallback explanation")
        self.assertEqual(service.client.models.calls, ["primary-test", "fallback-test"])

    def test_both_models_unavailable_returns_clean_503(self):
        service = AIService()
        service.client = RetryThenFallbackClient(fail_fallback=True)

        with (
            patch.object(settings, "GEMINI_MODEL", "primary-test"),
            patch.object(settings, "GEMINI_FALLBACK_MODEL", "fallback-test"),
        ):
            with self.assertRaises(AIServiceError) as raised:
                service._generate("test prompt")

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            raised.exception.detail,
            "AI generation is temporarily unavailable. Please try again later.",
        )
        self.assertEqual(service.client.models.calls, ["primary-test", "fallback-test"])

    def test_timeout_uses_fallback_once_and_returns_success(self):
        service = AIService()
        service.client = RetryThenFallbackClient(
            primary_error=TimeoutError("request timed out"),
        )

        with (
            patch.object(settings, "GEMINI_MODEL", "primary-test"),
            patch.object(settings, "GEMINI_FALLBACK_MODEL", "fallback-test"),
        ):
            response = service._generate("test prompt")

        self.assertEqual(response.text, "Fallback explanation")
        self.assertEqual(service.client.models.calls, ["primary-test", "fallback-test"])

    def test_timeout_without_fallback_returns_gateway_timeout(self):
        service = AIService()
        service.client = RetryThenFallbackClient(
            primary_error=TimeoutError("request timed out")
        )

        with (
            patch.object(settings, "GEMINI_MODEL", "primary-test"),
            patch.object(settings, "GEMINI_FALLBACK_MODEL", ""),
        ):
            with self.assertRaises(AIServiceError) as raised:
                service._generate("test prompt")

        self.assertEqual(raised.exception.status_code, 504)
        self.assertEqual(service.client.models.calls, ["primary-test"])

    def test_gemini_client_has_bounded_timeout_and_no_sdk_retries(self):
        with (
            patch.object(settings, "GEMINI_API_KEY", "test-key"),
            patch("backend.services.ai_service.genai.Client") as create_client,
        ):
            AIService()

        options = create_client.call_args.kwargs["http_options"]
        self.assertEqual(options.timeout, 15_000)
        self.assertEqual(options.retry_options.attempts, 1)

    def test_daily_quota_error_is_not_retried_or_sent_to_fallback(self):
        service = AIService()
        service.client = QuotaLimitClient()

        with (
            patch.object(settings, "GEMINI_MODEL", "primary-test"),
            patch.object(settings, "GEMINI_FALLBACK_MODEL", "fallback-test"),
        ):
            with self.assertRaises(AIServiceError) as raised:
                service._generate("test prompt")

        self.assertEqual(raised.exception.status_code, 429)
        self.assertIn("quota has been reached", raised.exception.detail)
        self.assertEqual(service.client.models.calls, ["primary-test"])


if __name__ == "__main__":
    unittest.main()
