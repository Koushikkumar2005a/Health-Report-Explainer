import asyncio
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from backend.rag import embeddings
from backend.rag.document_loader import clean_text, extract_pages, load_document
from backend.rag.rag_pipeline import retrieve_report_context
from backend.rag.retriever import (
    build_retrieval_queries,
    extract_medical_terms,
    retrieve_for_report,
    retrieve_query,
)
from backend.rag.text_chunker import chunk_text
from backend.rag.vector_store import (
    _reset_store_for_tests,
    collection_count,
    get_collection,
    get_indexed_source_hashes,
    query_vectors,
    remove_missing_sources,
    upsert_source_chunks,
)


class FakeEmbeddingModel:
    def __init__(self):
        self.inputs = []

    def encode(self, texts, **kwargs):
        self.inputs.extend(texts)
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in set(text.lower().split()):
                position = int(hashlib.sha256(token.encode()).hexdigest(), 16) % 768
                vectors[row, position] += 1
        if kwargs.get("normalize_embeddings", False):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1, norms)
        return vectors


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        prompt_data = json.loads(contents.rsplit("\n", 1)[-1])
        response_tests = []
        for test in prompt_data["tests"]:
            source_ids = [item["source_id"] for item in test["evidence"]]
            response_tests.append({
                "test_name": test["test_name"],
                "explanation": "A source-supported explanation." if source_ids else "",
                "source_ids": source_ids[:1],
            })
        return SimpleNamespace(text=json.dumps({"tests": response_tests}))


class RagPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_model = embeddings._model
        self.fake_model = FakeEmbeddingModel()
        embeddings._model = self.fake_model
        self.path_patch = patch("backend.rag.vector_store.CHROMA_PATH", Path(self.temp_dir.name))
        self.path_patch.start()
        _reset_store_for_tests()

    def tearDown(self):
        _reset_store_for_tests()
        embeddings._model = self.original_model
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_required_knowledge_directories_exist(self):
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "data/medical_knowledge/raw/CBC",
            "data/medical_knowledge/raw/Diabetes",
            "data/medical_knowledge/raw/LFT",
            "data/medical_knowledge/raw/KFT",
            "data/medical_knowledge/raw/Lipid_Profile",
            "data/medical_knowledge/raw/Thyroid",
            "data/medical_knowledge/processed",
            "data/chroma_db",
        ):
            self.assertTrue((root / relative).is_dir(), relative)

    def test_embedding_model_is_initialized_once(self):
        factory_calls = []

        def factory(name, **kwargs):
            factory_calls.append((name, kwargs))
            return self.fake_model

        with (
            patch.object(embeddings, "_model", None),
            patch.dict("sys.modules", {"sentence_transformers": SimpleNamespace(SentenceTransformer=factory)}),
        ):
            first = embeddings.get_embedding_model()
            second = embeddings.get_embedding_model()
        self.assertIs(first, second)
        self.assertEqual(
            factory_calls,
            [(
                "intfloat/multilingual-e5-base",
                {"revision": embeddings.MODEL_REVISION},
            )],
        )

    def test_embedding_prefix_dimension_and_normalization(self):
        passage = embeddings.embed_documents(["reference text"])
        query = embeddings.embed_queries(["search text"])
        self.assertTrue(self.fake_model.inputs[-2].startswith("passage: "))
        self.assertTrue(self.fake_model.inputs[-1].startswith("query: "))
        self.assertEqual(len(passage[0]), 768)
        self.assertAlmostEqual(float(np.linalg.norm(passage[0])), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(query[0])), 1.0, places=5)

    def test_text_cleaning_and_chunk_overlap(self):
        self.assertEqual(clean_text("  Alpha   beta\r\n\r\n\r\nGamma "), "Alpha beta\n\nGamma")
        chunks = chunk_text(("Alpha topic. " * 20).strip(), chunk_size=80, overlap=12)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(item) <= 80 for item in chunks))
        self.assertTrue(set(chunks[0].split()[-2:]) & set(chunks[1].split()[:3]))
        self.assertEqual(chunk_text("   "), [])

    def test_retrieval_queries_reflect_report_flags_without_inference(self):
        queries = build_retrieval_queries(
            ["haemoglobin"],
            "Hb: [reported value] [unit] LOW",
        )
        self.assertIn("What can a low haemoglobin result mean?", queries["haemoglobin"])
        self.assertFalse(any("high" in query for query in queries["haemoglobin"]))

    def test_txt_extraction_and_source_metadata(self):
        root = Path(self.temp_dir.name) / "raw"
        source = root / "CBC" / "reference.txt"
        source.parent.mkdir(parents=True)
        source.write_text("A neutral alpha reference passage.", encoding="utf-8")
        document = load_document(source, root)
        self.assertEqual(document.category, "CBC")
        self.assertEqual(document.source_filename, "reference.txt")
        self.assertEqual(document.pages[0].page_number, 1)
        self.assertEqual(document.pages[0].text, "A neutral alpha reference passage.")

    def test_pdf_extraction_uses_page_numbers(self):
        import pymupdf

        root = Path(self.temp_dir.name) / "raw"
        source = root / "Thyroid" / "reference.pdf"
        source.parent.mkdir(parents=True)
        pdf = pymupdf.open()
        pdf.new_page().insert_text((72, 72), "A neutral alpha reference passage.")
        pdf.new_page().insert_text((72, 72), "Another neutral beta passage.")
        pdf.save(source)
        pdf.close()

        pages = extract_pages(source)
        self.assertEqual([page.page_number for page in pages], [1, 2])
        self.assertIn("alpha reference", pages[0].text)

    def test_chroma_initialization_and_empty_collection(self):
        collection = get_collection()
        self.assertEqual(collection.name, "medical_knowledge")
        self.assertEqual(collection_count(), 0)
        result = retrieve_for_report("haemoglobin result")
        self.assertFalse(result["rag_used"])
        self.assertEqual(result["retrieval_status"], "insufficient_evidence")
        self.assertEqual(
            retrieve_report_context("haemoglobin result")["retrieval_status"],
            "empty_knowledge_base",
        )

    def test_recognition_runs_once_when_no_tests_are_detected(self):
        report = "Clinical history and examination are documented here."
        with (
            patch("backend.rag.rag_pipeline.collection_count", return_value=1),
            patch(
                "backend.rag.rag_pipeline.extract_medical_terms",
                wraps=extract_medical_terms,
            ) as recognize,
        ):
            result = retrieve_report_context(report)

        self.assertEqual(result["retrieval_status"], "no_tests_detected")
        self.assertEqual(result["terms"], [])
        self.assertEqual(result["tests"], [])
        recognize.assert_called_once_with(report)

    def test_recognition_runs_once_when_retrieval_fails(self):
        report = "Hemoglobin result is present."
        with (
            patch(
                "backend.rag.rag_pipeline.collection_count",
                side_effect=RuntimeError("database unavailable"),
            ),
            patch(
                "backend.rag.rag_pipeline.extract_medical_terms",
                wraps=extract_medical_terms,
            ) as recognize,
        ):
            result = retrieve_report_context(report)

        self.assertEqual(result["retrieval_status"], "unavailable")
        self.assertEqual(result["terms"], ["haemoglobin"])
        self.assertEqual(result["tests"][0]["test_name"], "haemoglobin")
        recognize.assert_called_once_with(report)

    def test_empty_knowledge_build_is_friendly(self):
        from scripts import build_knowledge_base

        with (
            patch.object(build_knowledge_base, "RAW_ROOT", Path(self.temp_dir.name)),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(build_knowledge_base.main(), 0)
        self.assertIn("No PDF or TXT medical reference documents were found", output.getvalue())

    def test_build_indexes_skips_unchanged_files_and_updates_changed_files(self):
        from scripts import build_knowledge_base

        raw_root = Path(self.temp_dir.name) / "raw"
        source = raw_root / "CBC" / "reference.txt"
        source.parent.mkdir(parents=True)
        source.write_text("A neutral alpha reference passage.", encoding="utf-8")
        with patch.object(build_knowledge_base, "RAW_ROOT", raw_root):
            with contextlib.redirect_stdout(io.StringIO()) as first_output:
                self.assertEqual(build_knowledge_base.main(), 0)
            self.assertIn("1 chunks indexed", first_output.getvalue())
            self.assertEqual(collection_count(), 1)

            with contextlib.redirect_stdout(io.StringIO()) as second_output:
                self.assertEqual(build_knowledge_base.main(), 0)
            self.assertIn("1 unchanged files skipped", second_output.getvalue())
            self.assertEqual(collection_count(), 1)

            source.write_text("A neutral beta reference passage.", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as updated_output:
                self.assertEqual(build_knowledge_base.main(), 0)
            self.assertIn("1 chunks indexed", updated_output.getvalue())
            records = get_collection().get(include=["documents", "metadatas"])
            self.assertEqual(collection_count(), 1)
            self.assertIn("beta", records["documents"][0])
            self.assertEqual(records["metadatas"][0]["category"], "CBC")

    def test_chroma_storage_retrieval_and_source_metadata(self):
        chunks = [
            {
                "id": "neutral-alpha-1",
                "text": "The alpha topic appears in this neutral reference passage.",
                "metadata": {
                    "source_path": "CBC/neutral.txt",
                    "source_filename": "neutral.txt",
                    "category": "CBC",
                    "page_number": 2,
                    "chunk_id": "2-1",
                    "source_hash": "version-one",
                },
            }
        ]
        self.assertEqual(upsert_source_chunks("CBC/neutral.txt", chunks), 1)
        self.assertEqual(collection_count(), 1)
        results = query_vectors(["alpha topic"], top_k=1)
        self.assertEqual(results[0]["metadata"]["source_filename"], "neutral.txt")
        self.assertEqual(results[0]["metadata"]["category"], "CBC")
        self.assertEqual(results[0]["metadata"]["page_number"], 2)
        self.assertEqual(results[0]["metadata"]["chunk_id"], "2-1")
        self.assertGreater(results[0]["relevance_score"], 0.0)

        _reset_store_for_tests()
        self.assertEqual(collection_count(), 1)
        self.assertEqual(get_collection().name, "medical_knowledge")

    def test_duplicate_updates_and_missing_source_cleanup(self):
        original = {
            "id": "stable-id",
            "text": "The alpha topic is represented by the first passage.",
            "metadata": {
                "source_path": "CBC/reference.txt",
                "source_filename": "reference.txt",
                "category": "CBC",
                "page_number": 1,
                "chunk_id": "1-1",
                "source_hash": "first-hash",
            },
        }
        updated = {
            **original,
            "text": "The beta topic replaces the first passage.",
            "metadata": {**original["metadata"], "source_hash": "second-hash"},
        }
        upsert_source_chunks("CBC/reference.txt", [original])
        upsert_source_chunks("CBC/reference.txt", [updated])
        self.assertEqual(collection_count(), 1)
        self.assertEqual(get_indexed_source_hashes()["CBC/reference.txt"], "second-hash")
        self.assertEqual(remove_missing_sources(set()), 1)
        self.assertEqual(collection_count(), 0)

    def test_category_filter_and_threshold(self):
        for record_id, category, text in (
            ("cbc-1", "CBC", "alpha topic reference"),
            ("thyroid-1", "Thyroid", "alpha topic explanation"),
        ):
            upsert_source_chunks(
                f"{category}/reference.txt",
                [{
                    "id": record_id,
                    "text": text,
                    "metadata": {
                        "source_path": f"{category}/reference.txt",
                        "source_filename": "reference.txt",
                        "category": category,
                        "page_number": 1,
                        "chunk_id": "1-1",
                        "source_hash": record_id,
                    },
                }],
            )
        results = retrieve_query("alpha topic", category="CBC", relevance_threshold=0)
        self.assertEqual([result["metadata"]["category"] for result in results], ["CBC"])
        self.assertEqual(retrieve_query("alpha topic", relevance_threshold=1.0), [])

    def test_gemini_receives_retrieved_context_and_preserves_language(self):
        from backend.services.ai_service import AIService

        service = AIService()
        client = SimpleNamespace(models=FakeModels())
        service.client = client
        source_id = "src-valid-hba1c"
        retrieved = {
            "terms": ["HbA1c"],
            "tests": [{
                "test_name": "HbA1c",
                "observations": ["HbA1c: [reported value] [unit]; laboratory reference range: [reported range]."],
                "passages": [{
                    "id": "test-vector-id",
                    "source_id": source_id,
                    "text": "Neutral reference passage for prompt integration.",
                    "metadata": {
                        "source_filename": "reference.txt",
                        "category": "Diabetes",
                        "page_number": 3,
                    },
                    "relevance_score": 0.8,
                }],
                "sources": [{
                    "source_id": source_id,
                    "test_name": "HbA1c",
                    "source_filename": "reference.txt",
                    "category": "Diabetes",
                    "page_number": 3,
                    "chunk_id": "3-1",
                    "relevance_score": 0.8,
                }],
                "status": "retrieved",
            }],
            "passages": [],
            "sources": [],
            "retrieval_status": "success",
            "rag_used": True,
        }
        with patch("backend.services.ai_service.retrieve_report_context", return_value=retrieved):
            result = asyncio.run(
                service.analyze_report(
                    "HbA1c: [reported value] [unit]; laboratory reference range: [reported range].",
                    "tamil",
                )
            )
        prompt = client.models.calls[0]["contents"]
        self.assertIn("Neutral reference passage for prompt integration.", prompt)
        self.assertIn("reference.txt", prompt)
        self.assertIn("laboratory reference range: [reported range]", prompt)
        config = client.models.calls[0]["config"]
        self.assertIn("untrusted data", config["system_instruction"])
        self.assertEqual(config["response_mime_type"], "application/json")
        self.assertEqual(result["language"], "tamil")
        self.assertTrue(result["rag_used"])
        self.assertEqual(result["retrieval_status"], "success")
        self.assertEqual(result["test_explanations"][0]["source_ids"], [source_id])
        self.assertEqual(result["retrieved_sources"][0]["page_number"], 3)

    def test_analyze_endpoint_upload_and_language_compatibility(self):
        from fastapi.testclient import TestClient

        from backend.main import app
        from backend.services.ai_service import ai_service
        from backend.services.document_service import document_service

        response_data = {
            "language": "tamil",
            "explanation": "A simple explanation.",
            "report_name": "report.pdf",
            "test_explanations": [{
                "test_name": "haemoglobin",
                "evidence_status": "supported",
                "explanation": "A supported explanation.",
                "source_ids": ["src-valid"],
                "sources": [],
            }],
            "retrieved_sources": [],
            "rag_used": False,
            "retrieval_status": "unavailable",
        }
        with (
            patch.object(document_service, "extract_text", new=AsyncMock(return_value="Report text with sufficient length for analysis.")),
            patch.object(ai_service, "analyze_report", new=AsyncMock(return_value=response_data)) as analyze,
        ):
            response = TestClient(app).post(
                "/api/analyze",
                files={"file": ("report.pdf", b"temporary test upload", "application/pdf")},
                data={"language": "tamil"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["explanation"], "A simple explanation.")
            self.assertEqual(
                response.json()["test_explanations"][0]["source_ids"],
                ["src-valid"],
            )
            self.assertFalse(response.json()["rag_used"])
            self.assertEqual(response.json()["retrieval_status"], "unavailable")
            analyze.assert_awaited_once()
            self.assertEqual(analyze.await_args.kwargs["language"], "tamil")
            self.assertEqual(collection_count(), 0)

        health = TestClient(app).get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "healthy")

    def test_empty_extracted_upload_returns_clear_error_without_gemini(self):
        from fastapi.testclient import TestClient

        from backend.main import app
        from backend.services.ai_service import ai_service
        from backend.services.document_service import document_service

        with (
            patch.object(
                document_service,
                "extract_text",
                new=AsyncMock(return_value=" \n "),
            ) as extract,
            patch.object(ai_service, "analyze_report", new=AsyncMock()) as analyze,
        ):
            response = TestClient(app).post(
                "/api/analyze",
                files={"file": ("empty.pdf", b"temporary test upload", "application/pdf")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("No text could be extracted", response.json()["detail"])
        extract.assert_awaited_once()
        analyze.assert_not_awaited()

    def test_gemini_failure_is_returned_as_an_api_error(self):
        from fastapi.testclient import TestClient

        from backend.main import app
        from backend.services.ai_service import AIServiceError, ai_service
        from backend.services.document_service import document_service

        with (
            patch.object(
                document_service,
                "extract_text",
                new=AsyncMock(return_value="A sufficiently long synthetic report text."),
            ),
            patch.object(
                ai_service,
                "analyze_report",
                new=AsyncMock(
                    side_effect=AIServiceError("Gemini failed.", status_code=503)
                ),
            ),
        ):
            response = TestClient(app).post(
                "/api/analyze",
                files={"file": ("report.pdf", b"temporary test upload", "application/pdf")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Gemini failed.")

    def test_docx_upload_remains_accepted_by_analysis_endpoint(self):
        from fastapi.testclient import TestClient

        from backend.main import app
        from backend.services.ai_service import ai_service
        from backend.services.document_service import document_service

        with (
            patch.object(
                document_service,
                "extract_text",
                new=AsyncMock(return_value="A sufficiently long extracted DOCX report text."),
            ) as extract,
            patch.object(
                ai_service,
                "analyze_report",
                new=AsyncMock(return_value={
                    "language": "english",
                    "explanation": "No evidence-supported interpretation.",
                    "report_name": "report.docx",
                    "retrieved_sources": [],
                    "rag_used": False,
                    "retrieval_status": "empty_knowledge_base",
                }),
            ),
        ):
            response = TestClient(app).post(
                "/api/analyze",
                files={"file": ("report.docx", b"temporary test upload", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )
        self.assertEqual(response.status_code, 200)
        extract.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
