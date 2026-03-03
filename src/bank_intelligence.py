from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


def analyze_bank_flows(bank_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Bank statement intelligence layer for Indian corporate accounts.

    Expects at minimum:
      - date (datetime or string)
      - amount (credits positive, debits negative)
      - counterparty (name or identifier, optional but recommended)
      - narration (optional)

    Outputs:
      - cash_deposit_ratio
      - round_tripping_score (heuristic)
      - top_counterparty_share
      - counterparty_concentration_hhi
      - related_party_transfer_share (best-effort using 'related' keyword)
    """
    if bank_df is None or bank_df.empty:
        return {
            "bank_cash_deposit_ratio": 0.0,
            "bank_round_tripping_score": 0.0,
            "bank_top_counterparty_share": 0.0,
            "bank_counterparty_hhi": 0.0,
            "bank_related_party_transfer_share": 0.0,
        }

    df = bank_df.copy()

    # Normalise columns where possible
    if "amount" not in df.columns:
        raise ValueError("Bank DataFrame must contain an 'amount' column.")

    # Attempt to identify cash vs non-cash deposits via narration keywords
    cash_like = pd.Series(False, index=df.index)
    if "narration" in df.columns:
        narr = df["narration"].astype(str).str.lower()
        cash_keywords = ["cash deposit", "cash dep", "by cash", "cash chq"]
        cash_like = narr.apply(lambda x: any(k in x for k in cash_keywords))

    credits = df[df["amount"] > 0]
    total_credits = float(credits["amount"].sum())
    cash_credits = float(credits.loc[cash_like, "amount"].sum()) if not credits.empty else 0.0
    bank_cash_deposit_ratio = (cash_credits / total_credits) if total_credits > 0 else 0.0

    # Counterparty aggregation
    if "counterparty" in df.columns:
        cp = df.copy()
        cp["counterparty"] = cp["counterparty"].astype(str).str.strip().replace({"": "UNKNOWN", "nan": "UNKNOWN"})
        by_cp = (
            cp.groupby("counterparty")["amount"]
            .sum()
            .abs()
            .reset_index()
            .rename(columns={"amount": "volume"})
        )
        total_volume = float(by_cp["volume"].sum()) if not by_cp.empty else 0.0
        if total_volume > 0:
            by_cp["share"] = by_cp["volume"] / total_volume
            top_counterparty_share = float(by_cp["share"].max())
            hhi = float((by_cp["share"] ** 2).sum())
        else:
            top_counterparty_share = 0.0
            hhi = 0.0
    else:
        total_volume = float(df["amount"].abs().sum())
        top_counterparty_share = 0.0
        hhi = 0.0

    # Round-tripping heuristic:
    # - Same counterparty appears with high total inflow and outflow over short windows.
    round_tripping_score = 0.0
    if "counterparty" in df.columns and "date" in df.columns:
        tmp = df.copy()
        if not np.issubdtype(tmp["date"].dtype, np.datetime64):
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp = tmp.dropna(subset=["date"])
        if not tmp.empty:
            tmp["month"] = tmp["date"].dt.to_period("M")
            grouped = (
                tmp.groupby(["counterparty", "month"])["amount"]
                .agg(["sum", "count"])
                .reset_index()
                .rename(columns={"sum": "net_amount"})
            )
            grouped["abs_net"] = grouped["net_amount"].abs()
            # High transaction count with small net suggests spin (in~=out)
            suspicious_months = grouped[(grouped["count"] >= 5) & (grouped["abs_net"] < grouped["abs_net"].median())]
            if not suspicious_months.empty:
                round_tripping_score = min(1.0, len(suspicious_months) / max(len(grouped), 1))

    # Related-party transfer share (best-effort, string-based in absence of a master)
    related_party_share = 0.0
    if "counterparty" in df.columns:
        cp = df.copy()
        cp["counterparty"] = cp["counterparty"].astype(str).str.lower()
        related_mask = cp["counterparty"].str.contains("related", na=False) | cp["counterparty"].str.contains(
            "group", na=False
        )
        related_volume = float(cp.loc[related_mask, "amount"].abs().sum())
        total_volume_all = float(cp["amount"].abs().sum())
        related_party_share = (related_volume / total_volume_all) if total_volume_all > 0 else 0.0

    return {
        "bank_cash_deposit_ratio": float(bank_cash_deposit_ratio),
        "bank_round_tripping_score": float(round_tripping_score),
        "bank_top_counterparty_share": float(top_counterparty_share),
        "bank_counterparty_hhi": float(hhi),
        "bank_total_txn_volume": float(total_volume),
        "bank_related_party_transfer_share": float(related_party_share),
    }

