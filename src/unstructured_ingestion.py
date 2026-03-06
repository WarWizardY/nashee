from pathlib import Path
from typing import Iterable, Dict, Any, List

from PyPDF2 import PdfReader

try:
    from .document_ai.layout_parser import parse_document_layouts
except ImportError:
    parse_document_layouts = None


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    texts: list[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


def _split_sentences(text: str) -> List[str]:
    # Very simple sentence splitter for prototype purposes
    raw = text.replace("\n", " ")
    parts = [s.strip() for s in raw.split(".") if s.strip()]
    return parts


def analyze_unstructured_pdfs(paths: Iterable[Path]) -> Dict[str, Any]:
    """
    Very lightweight, keyword-based risk extraction from unstructured PDFs
    (annual reports, legal notices, rating reports, sanction letters, etc.).

    This is a prototype placeholder for full Document AI + NLP.
    """
    combined_text = ""
    for p in paths:
        extracted = False
        if parse_document_layouts is not None:
            try:
                # Use Advanced Document AI (Table Transformer + EasyOCR layout parsing)
                print(f"[*] Running Advanced Document AI on {p.name}...")
                doc_text = parse_document_layouts(str(p))
                if doc_text.strip():
                    combined_text += "\n" + doc_text
                    extracted = True
            except Exception as e:
                print(f"[!] Advanced Document AI failed on {p.name}: {e}. Falling back to PyPDF2.")
        
        if not extracted:
            combined_text += "\n" + _extract_pdf_text(p)

    text_lower = combined_text.lower()
    sentences = _split_sentences(text_lower)

    # Naive keyword counts as proxy risk indicators with simple context rules
    risk_keywords = {
        "litigation": ["litigation", "suit filed", "court case", "arbitration"],
        "default": ["default", "overdue", "npa", "non-performing"],
        "pledge": ["pledge", "pledged shares", "encumbered"],
        "downgrade": ["rating downgrade", "downgraded", "negative outlook"],
    }

    scores: Dict[str, float] = {}
    total_hits = 0
    sample_sentences: Dict[str, List[str]] = {k: [] for k in risk_keywords}

    for key, words in risk_keywords.items():
        hits = 0
        for sent in sentences:
            if any(w in sent for w in words):
                # Simple negation handling: skip sentences like "no defaults" or "without any litigation"
                if "no " + key in sent or "without any " + key in sent:
                    continue
                hits += 1
                if len(sample_sentences[key]) < 3:
                    sample_sentences[key].append(sent.strip())
        scores[f"{key}_hits"] = hits
        total_hits += hits

    # Aggregate a simple litigation / document risk score between 0 and 1
    litigation_hits = scores.get("litigation_hits", 0) + scores.get("default_hits", 0)
    litigation_risk_score = min(1.0, litigation_hits / 10.0) if total_hits > 0 else 0.0

    # Severity tiers
    if litigation_risk_score == 0:
        severity = "NONE"
    elif litigation_risk_score < 0.3:
        severity = "LOW"
    elif litigation_risk_score < 0.7:
        severity = "MEDIUM"
    else:
        severity = "HIGH"

    return {
        "unstructured_text_length": len(text_lower),
        "unstructured_total_hits": total_hits,
        "litigation_risk_score": float(litigation_risk_score),
        "litigation_severity": severity,
        "litigation_sample_sentences": sample_sentences.get("litigation", []),
        "default_sample_sentences": sample_sentences.get("default", []),
        "pledge_sample_sentences": sample_sentences.get("pledge", []),
        "downgrade_sample_sentences": sample_sentences.get("downgrade", []),
        **scores,
    }

