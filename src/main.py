from pathlib import Path
from typing import Optional, List

import typer

from . import ingestion
from .application import Application, ApplicationStatus
from .cam_generator import generate_cam_docx
from .feature_store import log_application
from .risk_engine import build_risk_inputs_from_summary, simple_rule_based_decision, RISK_POLICY, POLICY_HASH
from .unstructured_ingestion import analyze_unstructured_pdfs
from .qualitative_inputs import score_qualitative_notes
from .advanced_credit import analyze_cibil_pdf, analyze_epfo_payroll, analyze_related_party_ledger
from .document_ai import extract_financial_fields_from_pdf, segment_pdf_sections
from .graph_analysis import build_transaction_graph, compute_graph_risk_scores, save_graph_image
from .anomaly import compute_gst_anomalies, compute_bank_anomalies
from .stress_test import run_stress_tests
from .research_agent import summarize_research
from .transformer_nlp import analyze_texts_with_transformer
from .loan_extractor import extract_sanction_loan_features
from .gst_reconciliation import reconcile_gstr2a_vs_3b
from .bank_intelligence import analyze_bank_flows
import uuid

app = typer.Typer(help="Intelli-Credit prototype CLI – runs a simple end-to-end CAM generation flow.")


@app.command()
def run_appraisal(
    company_name: str = typer.Option(..., help="Name of the borrowing company."),
    sector: str = typer.Option(..., help="Sector/industry of the company."),
    requested_limit: float = typer.Option(..., help="Requested loan limit (INR)."),
    gst_csv: Optional[Path] = typer.Option(None, help="Path to GST returns CSV (legacy prototype format)."),
    gstr2a_csv: Optional[Path] = typer.Option(None, "--gstr-2a-csv", help="Path to GSTR-2A CSV export."),
    gstr3b_csv: Optional[Path] = typer.Option(None, "--gstr-3b-csv", help="Path to GSTR-3B CSV export."),
    bank_csv: Optional[Path] = typer.Option(None, help="Path to bank statements CSV (prototype format)."),
    fin_csv: Optional[Path] = typer.Option(None, help="Path to ITR/financials CSV (prototype format)."),
    # Unstructured documents (multiple PDFs)
    unstructured_pdfs: Optional[List[Path]] = typer.Option(
        None,
        "--unstructured-pdf",
        help="Paths to unstructured document PDFs (annual reports, legal, rating, sanction letters).",
    ),
    # Qualitative notes
    officer_notes: Optional[List[str]] = typer.Option(
        None,
        "--note",
        help="Free-text qualitative notes from credit officer / site visits / management meetings.",
    ),
    # Advanced credit & compliance
    cibil_pdf: Optional[Path] = typer.Option(None, help="Path to CIBIL commercial report PDF."),
    epfo_csv: Optional[Path] = typer.Option(None, help="Path to EPFO / payroll CSV."),
    related_party_csv: Optional[Path] = typer.Option(
        None, help="Path to related party transactions ledger CSV."
    ),
    run_stress: bool = typer.Option(
        False,
        "--run-stress-tests",
        help="If set, run simple stress testing scenarios and log them in the application record.",
    ),
    output_docx: Path = typer.Option(
        Path("output") / "CAM_document.docx",
        help="Where to save the generated CAM DOCX.",
    ),
) -> None:
    """
    Run a minimal end-to-end credit appraisal:
    - Create an application record.
    - Load structured inputs (GST, bank, financials).
    - Load and analyze unstructured PDFs and advanced credit documents.
    - Ingest qualitative notes from credit officer.
    - Summarize key metrics, anomaly scores, graph risk and external signals.
    - Run a rule-based risk engine overlaying all signals.
    - Generate a CAM document in DOCX format.
    - Persist a feature/decision trace for reproducibility.
    """
    app_id = str(uuid.uuid4())
    application = Application(
        id=app_id,
        company_name=company_name,
        sector=sector,
        requested_limit=requested_limit,
        policy_hash=POLICY_HASH,
        policy_snapshot=RISK_POLICY,
    )

    # Legacy GST loader (single CSV)
    gst_df = ingestion.load_gst_returns(gst_csv) if gst_csv else None
    # New: separate 2A/3B inputs
    gstr2a_df = ingestion.load_gst_returns(gstr2a_csv) if gstr2a_csv else None
    gstr3b_df = ingestion.load_gst_returns(gstr3b_csv) if gstr3b_csv else None
    bank_df = ingestion.load_bank_statements(bank_csv) if bank_csv else None
    fin_df = ingestion.load_itr_financials(fin_csv) if fin_csv else None
    application.status = ApplicationStatus.INGESTED

    # Collect extra risk signals from all other input types
    extra_signals: dict[str, Any] = {}
    # GST reconciliation (GSTR-2A vs 3B) – India-specific intelligence layer
    if gstr2a_df is not None and gstr3b_df is not None:
        extra_signals.update(reconcile_gstr2a_vs_3b(gstr2a_df, gstr3b_df))
        # also record basic periods for completeness flags
        if "period" in gstr2a_df.columns:
            extra_signals["gstr2a_periods"] = int(gstr2a_df["period"].nunique())
        if "period" in gstr3b_df.columns:
            extra_signals["gstr3b_periods"] = int(gstr3b_df["period"].nunique())


    # Derive optional document_type hint from officer notes, e.g. "document_type=corporate_annual_report"
    document_type: str | None = None
    if officer_notes:
        for note in officer_notes:
            if "document_type=" in note:
                after = note.split("document_type=", 1)[1].strip()
                document_type = after.split()[0].strip().lower()
                break
    if document_type:
        extra_signals["document_type"] = document_type

    if unstructured_pdfs:
        # Unstructured risk keywords (legal, downgrade, etc.)
        extra_signals.update(analyze_unstructured_pdfs(unstructured_pdfs))

        # Document AI – attempt structured financial fields and section segmentation from the first large report
        first_pdf = unstructured_pdfs[0]
        fin_fields = extract_financial_fields_from_pdf(first_pdf)
        extra_signals.update(
            {
                "doc_ai_revenue": float(fin_fields.revenue or 0.0),
                "doc_ai_total_debt": float(fin_fields.total_debt or 0.0),
                "doc_ai_contingent_liabilities": float(fin_fields.contingent_liabilities or 0.0),
                "doc_ai_auditor_qualifications_present": fin_fields.auditor_qualifications_present,
            }
        )
        sections = segment_pdf_sections(first_pdf)
        extra_signals["doc_ai_sections_detected"] = list(sections.keys())

        # Sanction letter / loan feature extraction – only when document type is not corporate annual report
        if (document_type or "").lower() != "corporate_annual_report":
            for p in unstructured_pdfs:
                sanction_features = extract_sanction_loan_features(p)
                if sanction_features.get("sanction_loan_amount") is not None:
                    extra_signals.update(sanction_features)
                    break

    if officer_notes:
        extra_signals.update(score_qualitative_notes(officer_notes))

    if cibil_pdf:
        extra_signals.update(analyze_cibil_pdf(cibil_pdf))

    if epfo_csv:
        extra_signals.update(analyze_epfo_payroll(epfo_csv))

    if related_party_csv:
        extra_signals.update(analyze_related_party_ledger(related_party_csv))

    # Research agent – external news (Digital Credit Manager prototype)
    extra_signals.update(summarize_research(company_name=company_name, sector=sector))

    # Transformer NLP over combined qualitative + headline text (optional, if available)
    transformer_corpus: List[str] = []
    if officer_notes:
        transformer_corpus.extend(officer_notes)
    news_titles = extra_signals.get("research_news_titles") or []
    transformer_corpus.extend(news_titles)
    tf_signals = analyze_texts_with_transformer(transformer_corpus)
    if tf_signals:
        extra_signals.update(tf_signals)

    # Derive higher-level research sentiment features if transformer signals are available
    pos_share = extra_signals.get("transformer_pos_share")
    neg_share = extra_signals.get("transformer_neg_share")
    if pos_share is not None and neg_share is not None:
        try:
            pos_f = float(pos_share)
            neg_f = float(neg_share)
            extra_signals["news_sentiment_score"] = pos_f - neg_f
            extra_signals["promoter_risk_score"] = max(0.0, neg_f)
        except Exception:
            pass

    # Transaction graph analysis (GST + related party)
    if gst_csv or related_party_csv:
        G = build_transaction_graph(gst_csv=gst_csv, related_party_csv=related_party_csv)
        extra_signals.update(compute_graph_risk_scores(G))
        graph_image_path = Path("output") / f"graph_{app_id}.png"
        save_graph_image(G, graph_image_path)
        extra_signals["graph_image_path"] = str(graph_image_path)

    # Simple anomaly detection for GST and bank
    if gst_df is not None:
        extra_signals.update(compute_gst_anomalies(gst_df))
    if bank_df is not None:
        extra_signals.update(compute_bank_anomalies(bank_df))

        # Bank flow intelligence (cash ratio, round-tripping, counterparty concentration)
        extra_signals.update(analyze_bank_flows(bank_df))

    # Data completeness / confidence
    provided_sources = {
        "gst": gst_df is not None,
        "bank": bank_df is not None,
        "financials": fin_df is not None,
        "unstructured_pdfs": bool(unstructured_pdfs),
        "officer_notes": bool(officer_notes),
        "cibil": cibil_pdf is not None,
        "epfo": epfo_csv is not None,
        "related_party": related_party_csv is not None,
    }
    completeness_score = sum(1 for v in provided_sources.values() if v) / len(provided_sources)
    extra_signals["data_completeness_score"] = float(completeness_score)
    extra_signals["data_sources_present"] = [k for k, v in provided_sources.items() if v]

    summary = ingestion.summarize_inputs(gst_df, bank_df, fin_df, extra_signals=extra_signals)
    application.features = summary
    application.status = ApplicationStatus.ANALYZED

    risk_inputs = build_risk_inputs_from_summary(summary)
    decision = simple_rule_based_decision(risk_inputs, requested_limit=requested_limit, sector=sector)
    application.decision = {
        "approve": decision.approve,
        "recommended_limit": decision.recommended_limit,
        "recommended_rate": decision.recommended_rate,
        "score": decision.score,
        "risk_band": decision.risk_band,
        "pd_estimate": decision.pd_estimate,
        "reasons": decision.reasons,
    }

    if run_stress:
        application.decision["stress_tests"] = run_stress_tests(risk_inputs, requested_limit=requested_limit, sector=sector)
    application.status = ApplicationStatus.SCORED

    generate_cam_docx(
        output_path=output_docx,
        company_name=company_name,
        sector=sector,
        requested_limit=requested_limit,
        risk_decision=decision,
        input_summary=summary,
        application_id=application.id,
        engine_version=application.engine_version,
    )
    application.status = ApplicationStatus.CAM_GENERATED

    record_path = log_application(application)

    typer.echo(f"Application ID: {app_id}")
    typer.echo(f"CAM generated at: {output_docx}")
    typer.echo(f"Feature/decision trace stored at: {record_path}")


if __name__ == "__main__":
    app()

