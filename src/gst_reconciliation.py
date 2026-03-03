from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math

import pandas as pd


@dataclass
class GSTR2ANormalized:
    df: pd.DataFrame


@dataclass
class GSTR3BNormalized:
    df: pd.DataFrame


def _coalesce_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def normalize_gstr2a(df: pd.DataFrame) -> GSTR2ANormalized:
    """
    Normalize a wide range of common GSTR-2A exports to a standard schema.

    Required-ish outputs (best effort):
      - period (YYYY-MM or YYYY-MM-DD-like string)
      - supplier_gstin
      - invoice_no (optional)
      - taxable_value (float)
      - itc_amount (float)
    """
    df = df.copy()

    period_col = _coalesce_col(df, ["period", "month", "return_period", "fp"])
    supplier_col = _coalesce_col(df, ["supplier_gstin", "ctin", "seller_gstin", "gstin_supplier", "vendor_gstin"])
    invoice_col = _coalesce_col(df, ["invoice_no", "inum", "inv_no", "invoice_number"])
    taxable_col = _coalesce_col(df, ["taxable_value", "txval", "taxable", "taxable_amt"])
    itc_col = _coalesce_col(df, ["itc_amount", "itc", "itc_claimed", "itc_avail", "itc_availed", "igst+cgst+sgst"])

    out = pd.DataFrame()
    if period_col:
        out["period"] = df[period_col].astype(str)
    else:
        out["period"] = "UNKNOWN"

    if supplier_col:
        out["supplier_gstin"] = df[supplier_col].astype(str).str.strip()
    else:
        out["supplier_gstin"] = "UNKNOWN"

    if invoice_col:
        out["invoice_no"] = df[invoice_col].astype(str).str.strip()
    else:
        out["invoice_no"] = None

    def _to_num(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("INR", "", regex=False)
            .str.strip()
            .replace({"": "0", "nan": "0", "None": "0"})
            .astype(float)
        )

    out["taxable_value"] = _to_num(df[taxable_col]) if taxable_col else 0.0
    out["itc_amount"] = _to_num(df[itc_col]) if itc_col else 0.0

    # Clean obvious invalid GSTINs
    out["supplier_gstin"] = out["supplier_gstin"].fillna("UNKNOWN")

    return GSTR2ANormalized(df=out)


def normalize_gstr3b(df: pd.DataFrame) -> GSTR3BNormalized:
    """
    Normalize common GSTR-3B exports.

    We primarily need:
      - period
      - declared_supplies (proxy for revenue)
      - itc_claimed (if present)
      - output_tax_liability, cash_tax_paid (if present)
      - reverse_charge_turnover (if present)
      - refund_claimed, refund_sanctioned (if present)
    """
    df = df.copy()

    period_col = _coalesce_col(df, ["period", "month", "return_period", "fp"])
    declared_col = _coalesce_col(df, ["declared_supplies", "taxable_value", "gstr_3b_declared", "outward_supplies", "turnover"])
    itc_col = _coalesce_col(df, ["itc_claimed", "itc_availed", "itc", "gst_itc_claimed"])
    output_tax_col = _coalesce_col(df, ["output_tax_liability", "tax_payable", "output_tax", "gst_output_tax_liability"])
    cash_tax_col = _coalesce_col(df, ["cash_tax_paid", "tax_paid_cash", "paid_in_cash"])
    reverse_charge_col = _coalesce_col(df, ["reverse_charge_turnover", "reverse_charge", "rcm_turnover"])
    refund_claimed_col = _coalesce_col(df, ["refund_claimed", "refunds_claimed"])
    refund_sanctioned_col = _coalesce_col(df, ["refund_sanctioned", "refunds_sanctioned"])

    def _to_num(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("INR", "", regex=False)
            .str.strip()
            .replace({"": "0", "nan": "0", "None": "0"})
            .astype(float)
        )

    out = pd.DataFrame()
    out["period"] = df[period_col].astype(str) if period_col else "UNKNOWN"
    out["declared_supplies"] = _to_num(df[declared_col]) if declared_col else 0.0
    out["itc_claimed"] = _to_num(df[itc_col]) if itc_col else 0.0
    out["output_tax_liability"] = _to_num(df[output_tax_col]) if output_tax_col else 0.0
    out["cash_tax_paid"] = _to_num(df[cash_tax_col]) if cash_tax_col else 0.0
    out["reverse_charge_turnover"] = _to_num(df[reverse_charge_col]) if reverse_charge_col else 0.0
    out["refund_claimed"] = _to_num(df[refund_claimed_col]) if refund_claimed_col else 0.0
    out["refund_sanctioned"] = _to_num(df[refund_sanctioned_col]) if refund_sanctioned_col else 0.0

    return GSTR3BNormalized(df=out)


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def reconcile_gstr2a_vs_3b(
    gstr2a_raw: pd.DataFrame,
    gstr3b_raw: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Production-style reconciliation outputs (best-effort without requiring a specific vendor export):

    - ITC mismatch detection
    - Invoice-level variance aggregation (when invoice_no is present)
    - ITC concentration analysis (top supplier exposure, HHI)
    - Suspicious supplier heuristics (and optional clustering when sklearn is available)
    - Indian GST ratios (ITC dependency, refund intensity, reverse charge ratio, cash tax ratio)
    """
    g2a = normalize_gstr2a(gstr2a_raw).df
    g3b = normalize_gstr3b(gstr3b_raw).df

    itc_2a_total = float(g2a["itc_amount"].sum())
    itc_3b_total = float(g3b["itc_claimed"].sum())
    itc_variance = itc_3b_total - itc_2a_total
    itc_variance_ratio = abs(itc_variance) / max(itc_3b_total, 1.0)

    # Supplier concentration (2A ITC by supplier)
    by_supplier = (
        g2a.groupby("supplier_gstin", dropna=False)["itc_amount"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "itc_amount_sum", "count": "invoice_count"})
        .reset_index()
    )
    total_itc = max(float(by_supplier["itc_amount_sum"].sum()), 1.0)
    by_supplier["itc_share"] = by_supplier["itc_amount_sum"] / total_itc
    top_supplier_share = float(by_supplier["itc_share"].max()) if not by_supplier.empty else 0.0
    hhi = float((by_supplier["itc_share"] ** 2).sum()) if not by_supplier.empty else 0.0

    # Invoice-level variance aggregation (only if invoice_no exists)
    invoice_variance_total = 0.0
    if "invoice_no" in g2a.columns and g2a["invoice_no"].notna().any():
        inv = g2a.groupby(["supplier_gstin", "invoice_no"], dropna=False)["itc_amount"].sum().reset_index()
        invoice_variance_total = float(inv["itc_amount"].diff().abs().sum()) if len(inv) > 1 else 0.0

    # Suspicious suppliers heuristic: high share + low invoice count, or unusually high ITC per invoice
    by_supplier["itc_per_invoice"] = by_supplier["itc_amount_sum"] / by_supplier["invoice_count"].clip(lower=1)
    suspicious = by_supplier[
        (by_supplier["itc_share"] > 0.25)
        | ((by_supplier["itc_share"] > 0.10) & (by_supplier["invoice_count"] <= 2))
        | (by_supplier["itc_per_invoice"] > by_supplier["itc_per_invoice"].quantile(0.95) if len(by_supplier) >= 20 else False)
    ]
    suspicious_count = int(len(suspicious))
    suspicious_examples = suspicious.sort_values("itc_share", ascending=False)["supplier_gstin"].head(5).tolist()

    # Optional: clustering suppliers (best-effort)
    cluster_count = None
    try:
        from sklearn.cluster import KMeans  # type: ignore

        if len(by_supplier) >= 8:
            X = by_supplier[["itc_share", "invoice_count", "itc_per_invoice"]].fillna(0.0)
            k = min(4, max(2, int(math.sqrt(len(by_supplier)))))
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            by_supplier["cluster"] = labels
            cluster_count = int(k)
    except Exception:
        pass

    # Indian context ratios from 3B
    declared_supplies_total = float(g3b["declared_supplies"].sum())
    output_tax_total = float(g3b["output_tax_liability"].sum())
    cash_tax_total = float(g3b["cash_tax_paid"].sum())
    reverse_charge_turnover_total = float(g3b["reverse_charge_turnover"].sum())
    refund_claimed_total = float(g3b["refund_claimed"].sum())
    refund_sanctioned_total = float(g3b["refund_sanctioned"].sum())

    itc_dependency_ratio = _safe_div(itc_3b_total, output_tax_total)  # how dependent on ITC to discharge output tax
    cash_tax_ratio = _safe_div(cash_tax_total, output_tax_total)
    refund_intensity_ratio = _safe_div(refund_claimed_total, declared_supplies_total)
    refund_approval_ratio = _safe_div(refund_sanctioned_total, max(refund_claimed_total, 1.0))
    reverse_charge_turnover_ratio = _safe_div(reverse_charge_turnover_total, declared_supplies_total)

    return {
        # core reconciliation
        "gst_itc_total_2a": itc_2a_total,
        "gst_itc_total_3b": itc_3b_total,
        "gst_itc_variance": float(itc_variance),
        "gst_itc_variance_ratio": float(itc_variance_ratio),
        # concentration
        "gst_itc_top_supplier_share": float(top_supplier_share),
        "gst_itc_hhi": float(hhi),
        # suspicious vendors
        "gst_itc_suspicious_supplier_count": suspicious_count,
        "gst_itc_suspicious_supplier_examples": suspicious_examples,
        "gst_supplier_cluster_count": cluster_count,
        # invoice variance (proxy)
        "gst_invoice_variance_proxy": float(invoice_variance_total),
        # indian ratios
        "gst_itc_dependency_ratio": float(itc_dependency_ratio),
        "gst_cash_tax_ratio": float(cash_tax_ratio),
        "gst_refund_intensity_ratio": float(refund_intensity_ratio),
        "gst_refund_approval_ratio": float(refund_approval_ratio),
        "gst_reverse_charge_turnover_ratio": float(reverse_charge_turnover_ratio),
        # declared supplies proxy (for triangulation)
        "gst_declared_supplies_total": float(declared_supplies_total),
    }

