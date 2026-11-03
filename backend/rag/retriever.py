"""Medical-test recognition and test-scoped passage retrieval."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from typing import Any

from backend.rag.vector_store import query_vectors

logger = logging.getLogger(__name__)

TERM_ALIASES = {
    "haemoglobin": ("haemoglobin", "hemoglobin", "hgb", "hb"),
    "HbA1c": (
        "hba1c", "hb a1c", "a1c", "glycated haemoglobin", "glycated hemoglobin",
    ),
    "haematocrit": ("haematocrit", "hematocrit", "packed cell volume", "pcv", "hct"),
    "RBC count": ("red blood cell count", "red blood cells", "rbc count", "rbc"),
    "MCV": ("mean corpuscular volume", "mcv"),
    "MCH": ("mean corpuscular hemoglobin", "mean corpuscular haemoglobin", "mch"),
    "MCHC": ("mean corpuscular hemoglobin concentration", "mean corpuscular haemoglobin concentration", "mchc"),
    "RDW": ("red cell distribution width", "rdw"),
    "white blood cell count": ("white blood cell count", "white blood cells", "wbc count", "wbc", "leukocyte count"),
    "neutrophils": ("neutrophil", "neutrophils", "absolute neutrophil count", "anc"),
    "lymphocytes": ("lymphocyte", "lymphocytes"),
    "platelet count": ("platelet count", "platelets", "thrombocyte count", "thrombocytes"),
    "glucose": (
        "blood glucose", "blood sugar", "fasting blood sugar", "fasting glucose",
        "fasting plasma glucose", "random blood sugar", "postprandial blood sugar",
        "fbs", "fpg", "rbs", "ppbs", "glucose",
    ),
    "creatinine": (
        "serum creatinine", "s creatinine", "creatinine", "serum creat", "creat",
    ),
    "eGFR": (
        "estimated glomerular filtration rate", "glomerular filtration rate",
        "egfr", "e gfr", "gfr",
    ),
    "urea": (
        "blood urea nitrogen", "bun", "b u n", "blood urea", "serum urea",
        "s urea", "urea",
    ),
    "uric acid": ("uric acid", "urate"),
    "TSH": (
        "thyroid stimulating hormone", "thyroid-stimulating hormone", "thyrotropin",
        "tsh", "t s h",
    ),
    "T3": (
        "triiodothyronine", "free triiodothyronine", "total t3", "total t 3",
        "free t3", "free t 3", "ft3", "f t 3", "t3", "t 3",
    ),
    "T4": (
        "thyroxine", "tetraiodothyronine", "free thyroxine", "total t4",
        "total t 4", "free t4", "free t 4", "ft4", "f t 4", "t4", "t 4",
    ),
    "total cholesterol": ("total cholesterol", "serum cholesterol", "cholesterol total", "total chol", "tc"),
    "LDL cholesterol": ("low density lipoprotein", "low-density lipoprotein", "ldl cholesterol", "ldl-c", "ldl"),
    "HDL cholesterol": ("high density lipoprotein", "high-density lipoprotein", "hdl cholesterol", "hdl-c", "hdl"),
    "triglycerides": ("triglyceride", "triglycerides", "tg"),
    "ALT": ("alanine aminotransferase", "alanine transaminase", "sgpt", "alt"),
    "AST": ("aspartate aminotransferase", "aspartate transaminase", "sgot", "ast"),
    "alkaline phosphatase": ("alkaline phosphatase", "alp", "alk phos"),
    "bilirubin": ("bilirubin", "total bilirubin", "direct bilirubin"),
    "albumin": ("serum albumin", "albumin"),
    "total protein": ("total protein", "serum protein"),
    "ferritin": ("ferritin",),
    "serum iron": ("serum iron", "iron"),
    "vitamin D": ("25-hydroxyvitamin d", "25 hydroxy vitamin d", "25-oh vitamin d", "vitamin d", "25(oh)d"),
    "vitamin B12": ("vitamin b12", "vitamin b-12", "cobalamin", "b12"),
    "CRP": ("c-reactive protein", "crp"),
    "ESR": ("erythrocyte sedimentation rate", "sedimentation rate", "esr"),
}

TERM_CATEGORIES = {
    "haemoglobin": "CBC",
    "haematocrit": "CBC",
    "RBC count": "CBC",
    "MCV": "CBC",
    "MCH": "CBC",
    "MCHC": "CBC",
    "RDW": "CBC",
    "white blood cell count": "CBC",
    "neutrophils": "CBC",
    "lymphocytes": "CBC",
    "platelet count": "CBC",
    "HbA1c": "Diabetes",
    "glucose": "Diabetes",
    "creatinine": "KFT",
    "eGFR": "KFT",
    "urea": "KFT",
    "uric acid": "KFT",
    "TSH": "Thyroid",
    "T3": "Thyroid",
    "T4": "Thyroid",
    "total cholesterol": "Lipid_Profile",
    "LDL cholesterol": "Lipid_Profile",
    "HDL cholesterol": "Lipid_Profile",
    "triglycerides": "Lipid_Profile",
    "ALT": "LFT",
    "AST": "LFT",
    "alkaline phosphatase": "LFT",
    "bilirubin": "LFT",
    "albumin": "LFT",
    "total protein": "LFT",
    "ferritin": "CBC",
    "serum iron": "CBC",
    "vitamin D": None,
    "vitamin B12": None,
    "CRP": None,
    "ESR": "CBC",
}

_RESULT_VALUE_PATTERN = re.compile(
    r"^\s*(?:[<>]\s*)?(?:[+-]?\d|low\b|high\b|normal\b|positive\b|negative\b|"
    r"reactive\b|nonreactive\b|detected\b|not detected\b)",
    re.IGNORECASE,
)
_AMBIGUOUS_ALIASES = {
    "hb", "hgb", "pcv", "hct", "rbc", "mcv", "mch", "mchc", "rdw", "wbc",
    "anc", "fbs", "fpg", "rbs", "ppbs", "creat", "gfr", "egfr", "bun",
    "tsh", "t3", "t 3", "t4", "t 4", "ft3", "ft4", "tc", "ldl", "ldl c",
    "hdl", "hdl c", "tg", "alt", "ast", "alp", "crp", "esr", "b12", "iron",
    "glucose", "cholesterol", "albumin", "urea",
}


def _normalize_recognition_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _has_result_value(text: str, alias: str) -> bool:
    pattern = _alias_pattern(alias)
    for match in pattern.finditer(text):
        suffix = text[match.end():]
        if _RESULT_VALUE_PATTERN.search(suffix):
            return True
        if match.start() == 0 and re.match(
            r"\s+(?:reported\s+)?(?:result|value)\b",
            suffix,
        ):
            return True
    return False


def _alias_pattern(alias: str) -> re.Pattern[str]:
    if " " not in alias and len(alias) <= 5 and alias.isalnum():
        expression = r"\s*".join(re.escape(character) for character in alias)
    else:
        expression = re.escape(alias)
    return re.compile(rf"(?<!\w){expression}(?!\w)")


def _contains_alias(
    text: str,
    aliases: tuple[str, ...],
    *,
    require_result_context: bool = False,
) -> bool:
    normalized_text = _normalize_recognition_text(text)
    for alias in aliases:
        normalized_alias = _normalize_recognition_text(alias)
        if not normalized_alias:
            continue
        pattern = _alias_pattern(normalized_alias)
        if not pattern.search(normalized_text):
            continue
        if (
            require_result_context
            and normalized_alias in _AMBIGUOUS_ALIASES
            and not _has_result_value(normalized_text, normalized_alias)
        ):
            continue
        return True
    return False


def _mentions_test(text: str, term: str, *, report_text: bool = False) -> bool:
    lowered = _normalize_recognition_text(text)
    if term == "haemoglobin":
        lowered = re.sub(r"\bglycated\s+(?:haemoglobin|hemoglobin)\b", " ", lowered)
    if _contains_alias(
        lowered,
        TERM_ALIASES[term],
        require_result_context=report_text,
    ):
        return True
    if term == "total cholesterol":
        return (
            re.search(r"(?<!\w)cholesterol(?!\w)", lowered) is not None
            and not re.search(r"(?<!\w)(?:ldl|hdl)(?!\w)", lowered)
            and (
                not report_text
                or _has_result_value(lowered, "cholesterol")
            )
        )
    return False


def _recognition_candidates(report_text: str) -> list[str]:
    lines = [line.strip() for line in report_text.splitlines() if line.strip()]
    candidates = list(lines)
    label_fragments = []
    for line in lines:
        words = _normalize_recognition_text(line).split()
        result_start = next(
            (index for index, word in enumerate(words) if re.search(r"\d", word)),
            len(words),
        )
        label_fragments.append(" ".join(words[:result_start]))

    for width in (2, 3):
        for start in range(len(label_fragments) - width + 1):
            group = label_fragments[start:start + width]
            if all(group) and all(len(fragment.split()) <= 3 for fragment in group):
                candidates.append(" ".join(group))
    return candidates


def extract_medical_terms(report_text: str) -> list[str]:
    """Identify known test names and terms without interpreting patient results."""
    candidates = _recognition_candidates(report_text)
    terms = [
        term
        for term in TERM_ALIASES
        if any(_mentions_test(line, term, report_text=True) for line in candidates)
    ]
    logger.info(
        "Medical test recognition completed "
        "(recognized_tests=%d, test_names=%s).",
        len(terms),
        ",".join(terms) if terms else "none",
    )
    return terms


def _matching_report_lines(term: str, report_text: str) -> list[str]:
    return [
        line
        for line in _recognition_candidates(report_text)
        if _mentions_test(line, term, report_text=True)
    ]


def build_retrieval_queries(terms: list[str], report_text: str = "") -> dict[str, list[str]]:
    """Build independent definition/result queries for each detected test."""
    queries_by_term = {}
    for term in terms:
        queries = [f"What does the {term} test measure?"]
        matching_lines = _matching_report_lines(term, report_text)
        if any(re.search(r"\b(low|below|decreased|reduced)\b", line, re.IGNORECASE) for line in matching_lines):
            queries.append(f"What can a low {term} result mean?")
        if any(re.search(r"\b(high|above|elevated|increased)\b", line, re.IGNORECASE) for line in matching_lines):
            queries.append(f"What can a high {term} result mean?")
        queries_by_term[term] = list(dict.fromkeys(queries))
    return queries_by_term


def _stable_source_id(record_id: str) -> str:
    return "src_" + hashlib.sha256(record_id.encode("utf-8")).hexdigest()


def _test_result(term: str, report_text: str) -> dict[str, Any]:
    return {
        "test_name": term,
        "category": TERM_CATEGORIES.get(term),
        "observations": _matching_report_lines(term, report_text),
        "passages": [],
        "sources": [],
        "status": "insufficient_evidence",
    }


def empty_test_results(
    report_text: str,
    recognized_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Create explicit insufficient-evidence entries without accessing the vector store."""
    return [
        _test_result(term, report_text)
        for term in (
            recognized_terms
            if recognized_terms is not None
            else extract_medical_terms(report_text)
        )
    ]


def retrieve_query(
    query: str,
    top_k: int = 4,
    relevance_threshold: float = 0.25,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve passages for a standalone query, optionally restricted to one category."""
    records = query_vectors([query], top_k=top_k, category=category)
    return [
        record
        for record in records
        if record["relevance_score"] >= relevance_threshold
    ]


def retrieve_for_report(
    report_text: str,
    top_k: int = 4,
    relevance_threshold: float = 0.25,
    recognized_terms: list[str] | None = None,
) -> dict[str, Any]:
    """Retrieve and associate a bounded set of passages independently for each test."""
    terms = (
        recognized_terms
        if recognized_terms is not None
        else extract_medical_terms(report_text)
    )
    if not terms:
        return {
            "terms": [],
            "tests": [],
            "passages": [],
            "sources": [],
            "retrieval_status": "no_tests_detected",
            "rag_used": False,
        }

    tests = []
    all_passages = []
    for term, queries in build_retrieval_queries(terms, report_text).items():
        test = _test_result(term, report_text)
        records = query_vectors(
            queries,
            top_k=top_k,
            category=TERM_CATEGORIES.get(term),
        )
        best_by_id = {}
        for record in records:
            if record["relevance_score"] < relevance_threshold:
                continue
            if not _mentions_test(record["text"], term):
                continue
            current = best_by_id.get(record["id"])
            if current is None or record["relevance_score"] > current["relevance_score"]:
                best_by_id[record["id"]] = record

        passages = sorted(
            best_by_id.values(),
            key=lambda item: item["relevance_score"],
            reverse=True,
        )[:top_k]
        for passage in passages:
            passage["source_id"] = _stable_source_id(str(passage["id"]))
            metadata = passage.get("metadata") or {}
            test["sources"].append(
                {
                    "source_id": passage["source_id"],
                    "test_name": term,
                    "source_filename": metadata.get("source_filename", "Unknown source"),
                    "category": metadata.get("category", "Uncategorized"),
                    "page_number": metadata.get("page_number", 1),
                    "chunk_id": metadata.get("chunk_id", ""),
                    "relevance_score": round(passage["relevance_score"], 4),
                }
            )
        test["passages"] = passages
        if passages:
            test["status"] = "retrieved"
        tests.append(test)
        all_passages.extend(passages)

    all_sources = [source for test in tests for source in test["sources"]]
    has_evidence_for_every_test = all(test["passages"] for test in tests)
    return {
        "terms": terms,
        "tests": tests,
        "passages": all_passages,
        "sources": all_sources,
        "retrieval_status": "success" if has_evidence_for_every_test else "insufficient_evidence",
        "rag_used": bool(all_passages),
    }
