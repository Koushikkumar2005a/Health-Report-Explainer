import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image
from pytesseract import TesseractNotFoundError

from backend.services.document_service import DocumentService


class DocumentServiceDiagnosticsTests(unittest.TestCase):
    def assert_safe_extraction_log(self, captured, method, text):
        output = "\n".join(captured.output)
        self.assertIn(f"method={method}", output)
        self.assertIn(f"extracted_characters={len(text)}", output)
        self.assertIn(
            f"nonempty_lines={sum(bool(line.strip()) for line in text.splitlines())}",
            output,
        )
        self.assertNotIn("Synthetic Person", output)
        self.assertNotIn("6.2", output)

    def test_image_ocr_diagnostics_do_not_log_extracted_text(self):
        service = DocumentService()
        text = "Patient Name: Synthetic Person\nHbA1c: 6.2%"
        with (
            patch.object(service, "_extract_from_image", new=AsyncMock(return_value=text)),
            self.assertLogs("backend.services.document_service", level="INFO") as captured,
        ):
            result = asyncio.run(service.extract_text("unused.png", "report.png"))

        self.assertEqual(result, text)
        self.assert_safe_extraction_log(captured, "image_ocr", text)

    def test_image_uses_local_tesseract_without_calling_gemini(self):
        service = DocumentService()
        text = "Patient Name: Synthetic Person\nHbA1c: 6.2%"
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "report.png")
            Image.new("RGB", (1, 1)).save(image_path)
            with (
                patch(
                    "backend.services.document_service.pytesseract.image_to_string",
                    return_value=text,
                ) as ocr,
                patch.object(service, "client") as gemini_client,
            ):
                result = asyncio.run(
                    service.extract_text(image_path, "report.png")
                )

        self.assertEqual(result, text)
        ocr.assert_called_once()
        gemini_client.models.generate_content.assert_not_called()

    def test_image_with_no_ocr_text_returns_a_clear_error(self):
        service = DocumentService()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "report.png")
            Image.new("RGB", (1, 1)).save(image_path)
            with patch(
                "backend.services.document_service.pytesseract.image_to_string",
                return_value=" \n\t",
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "No readable text was found in the image",
                ):
                    asyncio.run(service.extract_text(image_path, "report.png"))

    def test_image_without_tesseract_returns_a_configuration_error(self):
        service = DocumentService()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "report.png")
            Image.new("RGB", (1, 1)).save(image_path)
            with patch(
                "backend.services.document_service.pytesseract.image_to_string",
                side_effect=TesseractNotFoundError,
            ):
                with self.assertRaisesRegex(ValueError, "not available on PATH"):
                    asyncio.run(service.extract_text(image_path, "report.png"))

    def test_docx_extraction_diagnostics_do_not_log_extracted_text(self):
        service = DocumentService()
        text = "Patient Name: Synthetic Person\nHbA1c: 6.2%"
        with (
            patch.object(service, "_extract_from_docx", new=AsyncMock(return_value=text)),
            self.assertLogs("backend.services.document_service", level="INFO") as captured,
        ):
            result = asyncio.run(service.extract_text("unused.docx", "report.docx"))

        self.assertEqual(result, text)
        self.assert_safe_extraction_log(captured, "docx_extraction", text)

    def test_pdf_text_extraction_diagnostics_do_not_log_extracted_text(self):
        service = DocumentService()
        page_text = (
            "Patient Name: Synthetic Person\nHbA1c: 6.2%\n"
            "Synthetic reference line with sufficient text to use the PDF text path."
        )
        with (
            patch(
                "backend.services.document_service.PdfReader",
                return_value=SimpleNamespace(
                    pages=[SimpleNamespace(extract_text=lambda: page_text)]
                ),
            ),
            self.assertLogs("backend.services.document_service", level="INFO") as captured,
        ):
            result = asyncio.run(service.extract_text("unused.pdf", "report.pdf"))

        self.assertIn(page_text, result)
        self.assert_safe_extraction_log(captured, "pdf_text_extraction", result)

    def test_scanned_pdf_ocr_diagnostics_do_not_log_extracted_text(self):
        service = DocumentService()
        text = "Patient Name: Synthetic Person\nHbA1c: 6.2%"
        with (
            patch(
                "backend.services.document_service.PdfReader",
                return_value=SimpleNamespace(
                    pages=[SimpleNamespace(extract_text=lambda: "")]
                ),
            ),
            patch.object(service, "_extract_scanned_pdf", new=AsyncMock(return_value=text)),
            self.assertLogs("backend.services.document_service", level="INFO") as captured,
        ):
            result = asyncio.run(service.extract_text("unused.pdf", "report.pdf"))

        self.assertEqual(result, text)
        self.assert_safe_extraction_log(captured, "ocr", text)


if __name__ == "__main__":
    unittest.main()
