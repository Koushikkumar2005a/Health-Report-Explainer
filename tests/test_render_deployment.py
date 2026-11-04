import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


class RenderDeploymentTests(unittest.TestCase):
    def test_docker_configuration_includes_index_but_excludes_secrets(self):
        project = Path(__file__).resolve().parents[1]
        dockerfile = (project / "Dockerfile").read_text(encoding="utf-8")
        dockerignore = (project / ".dockerignore").read_text(encoding="utf-8")

        self.assertIn("FROM python:3.11-slim", dockerfile)
        self.assertIn("backend.rag.vector_store import validate_knowledge_base", dockerfile)
        self.assertIn("--host 0.0.0.0 --port ${PORT}", dockerfile)
        self.assertIn("tesseract-ocr-eng", dockerfile)
        self.assertIn("intfloat/multilingual-e5-base", dockerfile)
        self.assertIn(".env", dockerignore)
        self.assertNotIn("chroma_db/", dockerignore)

    def test_startup_validates_knowledge_base_and_serves_health_and_frontend(self):
        with patch("backend.main.validate_knowledge_base_metadata") as validate:
            with TestClient(app) as client:
                health = client.get("/api/health")
                home = client.get("/")
                javascript = client.get("/js/app.js")

        validate.assert_called_once_with()
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "healthy")
        self.assertEqual(home.status_code, 200)
        self.assertIn("Health Report Explainer", home.text)
        self.assertEqual(javascript.status_code, 200)
        self.assertIn("function renderResults", javascript.text)


if __name__ == "__main__":
    unittest.main()
