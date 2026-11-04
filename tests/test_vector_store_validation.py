import unittest
from unittest.mock import patch

from backend.rag import vector_store


class FakeCollection:
    def __init__(self, count, categories):
        self._count = count
        self._categories = categories

    def count(self):
        return self._count

    def get(self, *, include):
        self.requested_include = include
        return {
            "metadatas": [{"category": category} for category in self._categories]
        }


class KnowledgeBaseValidationTests(unittest.TestCase):
    def test_validation_requires_nonempty_collection(self):
        collection = FakeCollection(0, [])
        with patch.object(vector_store, "get_collection", return_value=collection):
            with self.assertRaisesRegex(vector_store.VectorStoreError, "is empty"):
                vector_store.validate_knowledge_base()

    def test_validation_requires_all_reference_categories(self):
        collection = FakeCollection(100, ["CBC"])
        with patch.object(vector_store, "get_collection", return_value=collection):
            with self.assertRaisesRegex(
                vector_store.VectorStoreError,
                "missing categories",
            ):
                vector_store.validate_knowledge_base()

    def test_validation_checks_retrieval_for_each_category(self):
        categories = list(vector_store.REQUIRED_CATEGORY_QUERIES)
        collection = FakeCollection(658, categories)

        def fake_query(queries, *, top_k, category):
            return [{
                "text": f"Reference passage for {category}.",
                "metadata": {"category": category},
            }]

        with (
            patch.object(vector_store, "get_collection", return_value=collection),
            patch.object(vector_store, "query_vectors", side_effect=fake_query) as query,
        ):
            self.assertEqual(vector_store.validate_knowledge_base(), 658)

        self.assertEqual(query.call_count, len(categories))
        self.assertEqual(
            {call.kwargs["category"] for call in query.call_args_list},
            set(categories),
        )

    def test_validation_fails_if_category_retrieval_is_empty(self):
        categories = list(vector_store.REQUIRED_CATEGORY_QUERIES)
        collection = FakeCollection(658, categories)

        with (
            patch.object(vector_store, "get_collection", return_value=collection),
            patch.object(vector_store, "query_vectors", return_value=[]),
        ):
            with self.assertRaisesRegex(
                vector_store.VectorStoreError,
                "returned no passage",
            ):
                vector_store.validate_knowledge_base()


if __name__ == "__main__":
    unittest.main()
