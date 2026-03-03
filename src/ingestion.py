from pathlib import Path
from typing import Dict, Any

import pandas as pd


def load_gst_returns(file_path: Path) -> pd.DataFrame:
    """
    Minimal prototype loader for GST returns.
    Expects a CSV with columns like: period, gstin, counterparty_gstin, taxable_value, tax_amount.
    """
    return pd.read_csv(file_path)


def load_bank_statements(file_path: Path) -> pd.DataFrame:
    """
    Minimal prototype loader for bank statements.
    Expects a CSV with columns like: date, amount, balance, counterparty, narration.
    """
    return pd.read_csv(file_path, parse_dates=["date"])


def load_itr_financials(file_path: Path) -> pd.DataFrame:
    """
    Minimal prototype loader for ITR / financials.
    Expects a CSV with columns like: year, revenue, ebitda, pat, net_worth, total_debt.
    """
    return pd.read_csv(file_path)


def summarize_inputs(
    gst_df: pd.DataFrame | None,
    bank_df: pd.DataFrame | None,
    fin_df: pd.DataFrame | None,
    extra_signals: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Produce a simple, human-readable summary of core inputs for use in CAM.
    """
    summary: Dict[str, Any] = {}

    if gst_df is not None and not gst_df.empty:
        summary["gst_periods"] = gst_df["period"].nunique() if "period" in gst_df.columns else None
        summary["gst_total_taxable"] = float(gst_df.get("taxable_value", pd.Series(dtype=float)).sum())

    if bank_df is not None and not bank_df.empty:
        summary["bank_months"] = bank_df["date"].dt.to_period("M").nunique() if "date" in bank_df.columns else None
        summary["bank_total_inflows"] = float(bank_df[bank_df["amount"] > 0]["amount"].sum()) if "amount" in bank_df.columns else None
        summary["bank_total_outflows"] = float(bank_df[bank_df["amount"] < 0]["amount"].sum()) if "amount" in bank_df.columns else None

    if fin_df is not None and not fin_df.empty:
        latest = fin_df.sort_values("year").iloc[-1]
        summary["latest_year"] = int(latest.get("year", 0))
        summary["latest_revenue"] = float(latest.get("revenue", 0.0))
        summary["latest_ebitda"] = float(latest.get("ebitda", 0.0))
        summary["latest_pat"] = float(latest.get("pat", 0.0))
        summary["latest_net_worth"] = float(latest.get("net_worth", 0.0))
        summary["latest_total_debt"] = float(latest.get("total_debt", 0.0))

    # Cross-source consistency checks
    latest_revenue = summary.get("latest_revenue")
    gst_total = summary.get("gst_total_taxable")
    bank_inflows = summary.get("bank_total_inflows")

    if latest_revenue and gst_total:
        ratio = gst_total / latest_revenue if latest_revenue else None
        summary["gst_vs_itr_revenue_ratio"] = float(ratio) if ratio is not None else None

    if latest_revenue and bank_inflows:
        ratio = bank_inflows / latest_revenue if latest_revenue else None
        summary["bank_inflows_vs_revenue_ratio"] = float(ratio) if ratio is not None else None

    if extra_signals:
        summary.update(extra_signals)

    # Backward/forward-compatible GST presence flag: accept gstr2a/gstr3b periods too
    if summary.get("gst_periods") is None:
        if summary.get("gstr3b_periods") is not None:
            summary["gst_periods"] = summary.get("gstr3b_periods")

    # Flag whether we found any financials at all (CSV or Document AI)
    financials_found = False
    if summary.get("latest_revenue") or summary.get("latest_total_debt") or summary.get("latest_net_worth"):
        financials_found = True
    if summary.get("doc_ai_revenue") or summary.get("doc_ai_total_debt"):
        financials_found = True
    summary["financials_found_flag"] = financials_found

    return summary

