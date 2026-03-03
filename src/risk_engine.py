from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import hashlib
import json


def load_risk_policy(path: Path | None = None) -> Dict[str, Any]:
    """
    Load risk policy configuration from JSON so that
    thresholds and weights are configurable without code changes.
    """
    policy_path = path or Path("risk_policy.json")
    if not policy_path.exists():
        return {}
    with policy_path.open("r", encoding="utf-8") as f:
        return json.load(f)


RISK_POLICY: Dict[str, Any] = load_risk_policy()
POLICY_HASH: str = hashlib.sha256(json.dumps(RISK_POLICY, sort_keys=True).encode("utf-8")).hexdigest()


def get_effective_policy(sector: str | None) -> Dict[str, Any]:
    """
    Return a policy dict with optional sector-specific overrides applied.
    """
    base = dict(RISK_POLICY)
    sector = (sector or "").lower()
    sector_policies = RISK_POLICY.get("sector_policies", {})
    override = sector_policies.get(sector)
    if override:
        # shallow merge for prototype; nested dicts like "leverage" are merged individually
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                merged = dict(base[k])
                merged.update(v)
                base[k] = merged
            else:
                base[k] = v
    return base


@dataclass
class RiskInputs:
    """
    Aggregated features needed by the prototype risk engine.
    """

    latest_revenue: float
    latest_ebitda: float
    latest_net_worth: float
    latest_total_debt: float
    bank_total_inflows: float
    bank_total_outflows: float
    litigation_risk_score: float = 0.0
    management_quality_score: float = 0.5
    capacity_utilization_penalty: float = 0.0
    cibil_risk_score: float = 0.0
    payroll_stability_score: float = 0.5
    related_party_risk_score: float = 0.0
    graph_risk_score: float = 0.0
    data_completeness_score: float = 0.0
    has_gst: bool = False
    has_bank: bool = False
    # Sanction letter / existing loan features
    sanction_existing_debt: float = 0.0
    sanction_effective_rate: float = 0.0
    sanction_microfinance_exposure_flag: bool = False
    sanction_group_liability_flag: bool = False
    sanction_short_tenure_flag: bool = False
    sanction_high_interest_flag: bool = False
    # Research / anomaly features
    news_sentiment_score: float = 0.0
    promoter_risk_score: float = 0.0
    research_litigation_news_count: float = 0.0
    research_sector_headwind_score: float = 0.0
    gst_anomaly_score: float = 0.0
    bank_anomaly_score: float = 0.0
    financials_found_flag: bool = False
    # India-specific GST reconciliation features
    gst_itc_variance_ratio: float = 0.0
    gst_itc_top_supplier_share: float = 0.0
    gst_itc_dependency_ratio: float = 0.0
    gst_cash_tax_ratio: float = 0.0
    gst_reverse_charge_turnover_ratio: float = 0.0
    # Bank intelligence
    bank_cash_deposit_ratio: float = 0.0
    bank_round_tripping_score: float = 0.0
    bank_top_counterparty_share: float = 0.0
    bank_counterparty_hhi: float = 0.0
    bank_total_txn_volume: float = 0.0
    bank_related_party_transfer_share: float = 0.0


@dataclass
class RiskDecision:
    approve: bool
    recommended_limit: float
    recommended_rate: float
    score: float
    reasons: list[str]
    risk_band: str
    pd_estimate: float
    capacity_score: float
    character_score: float
    capital_score: float
    conditions_score: float
    collateral_score: float


def simple_rule_based_decision(
    features: RiskInputs,
    requested_limit: float,
    sector: str | None = None,
    base_rate: float | None = None,
) -> RiskDecision:
    """
    Very simple, transparent rule-based prototype that approximates
    the behavior of the final XGBoost + SHAP model.
    This is intentionally lightweight for hackathon/demo purposes.
    """
    reasons: list[str] = []

    policy = get_effective_policy(sector)
    base_rate = base_rate if base_rate is not None else float(policy.get("base_rate", 10.0))

    # --- 5C sub-scores initialisation (0–1, start at neutral 0.5) ---
    capacity_score = 0.5
    character_score = 0.5
    capital_score = 0.5
    conditions_score = 0.5
    collateral_score = 0.5

    # Basic leverage and coverage style heuristics (policy-driven)
    total_debt_for_leverage = features.latest_total_debt + features.sanction_existing_debt
    leverage = None
    revenue_to_limit = None
    if features.latest_net_worth > 0:
        leverage = (total_debt_for_leverage + requested_limit) / max(features.latest_net_worth, 1.0)
    if requested_limit > 0 and features.latest_revenue > 0:
        revenue_to_limit = features.latest_revenue / max(requested_limit, 1.0)
    lev_cfg = policy.get("leverage", {})
    rev_cfg = policy.get("revenue_to_limit", {})
    ebitda_cfg = policy.get("ebitda_margin", {})
    overlays = policy.get("overlays", {})

    # Net worth and leverage contribution → Capital
    lev_low = float(lev_cfg.get("low_threshold", 2.0))
    lev_med = float(lev_cfg.get("medium_threshold", 3.0))
    lev_w = lev_cfg.get("weights", {})
    if leverage is not None:
        if leverage <= lev_low:
            capital_score += float(lev_w.get("low", 0.4))
            reasons.append(f"Comfortable leverage (Debt/Net Worth <= {lev_low}x).")
        elif leverage <= lev_med:
            capital_score += float(lev_w.get("medium", 0.2))
            reasons.append(f"Moderate leverage (Debt/Net Worth between {lev_low}x and {lev_med}x).")
        else:
            capital_score += float(lev_w.get("high", -0.3))
            reasons.append(f"High leverage (Debt/Net Worth > {lev_med}x).")

    # Revenue vs requested limit → Capacity
    rev_high = float(rev_cfg.get("high_threshold", 4.0))
    rev_med = float(rev_cfg.get("medium_threshold", 2.0))
    rev_w = rev_cfg.get("weights", {})
    if revenue_to_limit is not None:
        if revenue_to_limit >= rev_high:
            capacity_score += float(rev_w.get("high", 0.3))
            reasons.append(
                f"Requested limit is conservative relative to revenue (Revenue / Limit >= {rev_high}x)."
            )
        elif revenue_to_limit >= rev_med:
            capacity_score += float(rev_w.get("medium", 0.1))
            reasons.append(
                f"Requested limit is reasonable relative to revenue (Revenue / Limit between {rev_med}x and {rev_high}x)."
            )
        else:
            capacity_score += float(rev_w.get("low", -0.2))
            reasons.append(f"Requested limit is aggressive relative to revenue (Revenue / Limit < {rev_med}x).")
    elif not features.financials_found_flag:
        reasons.append("Financial statements could not be extracted – capacity/capital assessed with low confidence.")

    # EBITDA comfort → Capacity
    ebitda_high = float(ebitda_cfg.get("high_threshold", 0.15))
    ebitda_med = float(ebitda_cfg.get("medium_threshold", 0.08))
    ebitda_w = ebitda_cfg.get("weights", {})
    if features.latest_revenue > 0:
        ebitda_margin = features.latest_ebitda / max(features.latest_revenue, 1.0)
        if ebitda_margin >= ebitda_high:
            capacity_score += float(ebitda_w.get("high", 0.2))
            reasons.append(f"Healthy EBITDA margin (>= {ebitda_high:.0%}).")
        elif ebitda_margin >= ebitda_med:
            capacity_score += float(ebitda_w.get("medium", 0.05))
            reasons.append(f"Acceptable EBITDA margin ({ebitda_med:.0%}–{ebitda_high:.0%}).")
        else:
            capacity_score += float(ebitda_w.get("low", -0.15))
            reasons.append(f"Thin EBITDA margin (< {ebitda_med:.0%}).")

    # Qualitative management / capacity overlay → Character & Capacity
    if features.management_quality_score >= 0.7:
        character_score += float(overlays.get("management_quality", 0.1))
        reasons.append("Positive qualitative comfort on management quality.")
    elif features.management_quality_score <= 0.3:
        character_score += float(overlays.get("management_concern", -0.1))
        reasons.append("Concerns from qualitative assessment of management.")

    if features.capacity_utilization_penalty > 0.0:
        capacity_score += float(overlays.get("capacity_penalty_base", -0.05)) * (1.0 + features.capacity_utilization_penalty)
        reasons.append("Observed suboptimal capacity utilization at plant.")

    # External risk overlays (CIBIL, litigation, related parties, payroll) → Character
    if features.cibil_risk_score > 0.0:
        character_score += float(overlays.get("cibil_factor", -0.2)) * features.cibil_risk_score
        reasons.append("Adverse elements in CIBIL commercial report.")

    if features.litigation_risk_score > 0.0:
        character_score += float(overlays.get("litigation_factor", -0.15)) * features.litigation_risk_score
        reasons.append("Elevated litigation / document-based risk signals.")

    if features.related_party_risk_score > 0.0:
        character_score += float(overlays.get("related_party_factor", -0.1)) * features.related_party_risk_score
        reasons.append("High concentration of related party transactions.")

    if features.graph_risk_score > 0.0:
        character_score += float(overlays.get("graph_factor", -0.15)) * features.graph_risk_score
        reasons.append("Transaction graph shows signs of circular trading / concentration.")

    if features.payroll_stability_score >= 0.8:
        character_score += float(overlays.get("payroll_positive", 0.05))
        reasons.append("Stable payroll / EPFO contributions over time.")
    elif features.payroll_stability_score <= 0.4:
        character_score += float(overlays.get("payroll_negative", -0.05))
        reasons.append("Limited visibility on payroll stability.")

    # Sanction-letter specific behavioral overlays
    if features.sanction_existing_debt > 0 and features.latest_total_debt == 0:
        reasons.append("Existing loan detected from sanction letter despite no financials provided.")

    if features.sanction_microfinance_exposure_flag:
        character_score += float(overlays.get("microfinance_exposure_factor", -0.05))
        reasons.append("Exposure to microfinance / JLG-type borrowing.")

    if features.sanction_group_liability_flag:
        character_score += float(overlays.get("group_liability_factor", -0.03))
        reasons.append("Group liability structure detected (JLG).")

    if features.sanction_short_tenure_flag:
        conditions_score += float(overlays.get("short_tenure_factor", -0.02))
        reasons.append("Short-tenure loan profile identified from sanction letter.")

    if features.sanction_high_interest_flag:
        character_score += float(overlays.get("high_interest_factor", -0.05))
        reasons.append("High interest rate detected on existing borrowing.")

    # Research / anomaly overlays
    if features.news_sentiment_score != 0.0:
        character_score += float(overlays.get("news_sentiment_factor", 0.05)) * features.news_sentiment_score
        reasons.append("External news sentiment incorporated into character assessment.")

    if features.promoter_risk_score > 0.0:
        character_score += float(overlays.get("promoter_risk_factor", -0.05)) * features.promoter_risk_score
        reasons.append("Promoter / qualitative risk signalled by negative sentiment.")

    if features.research_sector_headwind_score > 0.0:
        conditions_score += float(overlays.get("sector_headwind_factor", -0.05)) * features.research_sector_headwind_score
        reasons.append("Sector headwinds identified in external research.")

    if features.gst_anomaly_score > 0.0:
        conditions_score += float(overlays.get("gst_anomaly_factor", -0.05)) * features.gst_anomaly_score
        reasons.append("GST flow anomalies detected.")

    if features.bank_anomaly_score > 0.0:
        conditions_score += float(overlays.get("bank_anomaly_factor", -0.05)) * features.bank_anomaly_score
        reasons.append("Bank flow anomalies detected.")

    # India-specific GST reconciliation overlays
    if features.gst_itc_variance_ratio > 0.0:
        # High mismatch between 2A and 3B is a red flag
        conditions_score -= min(0.08, features.gst_itc_variance_ratio * 0.08)
        reasons.append("GSTR-2A vs 3B mismatch observed (ITC reconciliation variance).")

    if features.gst_itc_top_supplier_share > 0.0:
        if features.gst_itc_top_supplier_share > 0.35:
            character_score -= 0.04
            reasons.append("High ITC concentration to a small set of suppliers.")

    if features.gst_itc_dependency_ratio > 0.0:
        if features.gst_itc_dependency_ratio > 0.9:
            conditions_score -= 0.03
            reasons.append("High dependency on ITC to discharge output tax.")

    if features.gst_cash_tax_ratio > 0.0:
        if features.gst_cash_tax_ratio < 0.15:
            conditions_score -= 0.02
            reasons.append("Low cash tax payment ratio (higher reliance on ITC).")

    if features.gst_reverse_charge_turnover_ratio > 0.0:
        if features.gst_reverse_charge_turnover_ratio > 0.2:
            conditions_score -= 0.02
            reasons.append("Elevated reverse-charge turnover ratio flagged.")

    # Bank intelligence overlays
    if features.bank_cash_deposit_ratio > 0.0:
        if features.bank_cash_deposit_ratio > 0.4:
            conditions_score -= 0.03
            reasons.append("High cash deposit ratio observed in bank statements.")

    if features.bank_round_tripping_score > 0.0:
        if features.bank_round_tripping_score > 0.3:
            conditions_score -= 0.04
            reasons.append("Round-tripping patterns suspected from bank inflow/outflow loops.")

    if features.bank_top_counterparty_share > 0.0:
        if features.bank_top_counterparty_share > 0.35:
            character_score -= 0.03
            reasons.append("High dependence on a small number of banking counterparties.")

    if features.bank_related_party_transfer_share > 0.0:
        if features.bank_related_party_transfer_share > 0.25:
            character_score -= 0.03
            reasons.append("Significant share of bank flows appear to be related-party transfers.")

    # Ensure 5C scores are within [0, 1]
    def _clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    capacity_score = _clamp(capacity_score)
    character_score = _clamp(character_score)
    capital_score = _clamp(capital_score)
    conditions_score = _clamp(conditions_score)
    collateral_score = _clamp(collateral_score)

    five_c_weights = policy.get("five_c_weights", {})
    w_capacity = float(five_c_weights.get("capacity", 0.3))
    w_character = float(five_c_weights.get("character", 0.25))
    w_capital = float(five_c_weights.get("capital", 0.2))
    w_conditions = float(five_c_weights.get("conditions", 0.15))
    w_collateral = float(five_c_weights.get("collateral", 0.1))

    normalized_score = (
        capacity_score * w_capacity
        + character_score * w_character
        + capital_score * w_capital
        + conditions_score * w_conditions
        + collateral_score * w_collateral
    )

    # Penalize incomplete data (especially missing GST/bank)
    if not features.has_gst or not features.has_bank:
        normalized_score = max(0.0, normalized_score - 0.1)
        conditions_score = _clamp(conditions_score - 0.05)
        reasons.append("Key data sources (GST or bank) missing – applying prudential penalty.")
    if features.data_completeness_score < 0.5:
        normalized_score = max(0.0, normalized_score - 0.1)
        conditions_score = _clamp(conditions_score - 0.05)
        reasons.append("Low overall data completeness – conservative adjustment applied.")

    # Decision logic
    approve = normalized_score >= 0.5

    # Recommended limit scaling (with cap under incomplete data)
    if approve:
        # Scale requested limit up/down slightly based on score
        if normalized_score >= 0.75:
            recommended_limit = requested_limit * 1.1
        elif normalized_score >= 0.6:
            recommended_limit = requested_limit
        else:
            recommended_limit = requested_limit * 0.8

        # Cap limit if data is incomplete
        if features.data_completeness_score < 0.5:
            recommended_limit = min(recommended_limit, requested_limit * 0.7)
            reasons.append("Limit capped due to incomplete data.")
    else:
        recommended_limit = 0.0

    # Simple risk premium logic (policy-driven) + coarse bands/PD proxy
    bands = policy.get("spread_bands", {})
    strong_min = float(bands.get("strong_min", 0.75))
    moderate_min = float(bands.get("moderate_min", 0.6))
    borderline_min = float(bands.get("borderline_min", 0.5))
    spreads_cfg = bands.get("spreads", {})
    risk_band = "HIGH"
    pd_estimate = 0.2
    if normalized_score >= strong_min:
        spread = float(spreads_cfg.get("strong", 1.0))
        risk_band = "LOW"
        pd_estimate = 0.01
        reasons.append("Strong overall financial profile – pricing near base rate.")
    elif normalized_score >= moderate_min:
        spread = float(spreads_cfg.get("moderate", 2.0))
        risk_band = "MEDIUM"
        pd_estimate = 0.05
        reasons.append("Moderate risk – standard risk premium applied.")
    elif normalized_score >= borderline_min:
        spread = float(spreads_cfg.get("borderline", 3.0))
        risk_band = "ELEVATED"
        pd_estimate = 0.10
        reasons.append("Borderline acceptable – higher risk premium required.")
    else:
        spread = 0.0
        reasons.append("Score below approval threshold – facility should be rejected.")

    recommended_rate = base_rate + spread if approve else 0.0

    return RiskDecision(
        approve=approve,
        recommended_limit=recommended_limit,
        recommended_rate=recommended_rate,
        score=normalized_score,
        reasons=reasons,
        risk_band=risk_band,
        pd_estimate=pd_estimate,
        capacity_score=capacity_score,
        character_score=character_score,
        capital_score=capital_score,
        conditions_score=conditions_score,
        collateral_score=collateral_score,
    )


def build_risk_inputs_from_summary(summary: Dict[str, Any]) -> RiskInputs:
    """
    Helper to convert the ingestion summary dict into RiskInputs
    needed by the simple rule-based decision engine.
    """
    return RiskInputs(
        latest_revenue=float(summary.get("latest_revenue", 0.0)),
        latest_ebitda=float(summary.get("latest_ebitda", 0.0)),
        latest_net_worth=float(summary.get("latest_net_worth", 0.0)),
        latest_total_debt=float(summary.get("latest_total_debt", 0.0)),
        bank_total_inflows=float(summary.get("bank_total_inflows", 0.0)),
        bank_total_outflows=float(summary.get("bank_total_outflows", 0.0)),
        litigation_risk_score=float(summary.get("litigation_risk_score", 0.0)),
        management_quality_score=float(summary.get("management_quality_score", 0.5)),
        capacity_utilization_penalty=float(summary.get("capacity_utilization_penalty", 0.0)),
        cibil_risk_score=float(summary.get("cibil_risk_score", 0.0)),
        payroll_stability_score=float(summary.get("payroll_stability_score", 0.5)),
        related_party_risk_score=float(summary.get("related_party_risk_score", 0.0)),
        graph_risk_score=float(summary.get("graph_risk_score", 0.0)),
        data_completeness_score=float(summary.get("data_completeness_score", 0.0)),
        has_gst=bool(summary.get("gst_periods")),
        has_bank=bool(summary.get("bank_months")),
        sanction_existing_debt=float(summary.get("sanction_existing_debt", 0.0)),
        sanction_effective_rate=float(summary.get("sanction_effective_rate", 0.0)),
        sanction_microfinance_exposure_flag=bool(summary.get("sanction_microfinance_exposure_flag", False)),
        sanction_group_liability_flag=bool(summary.get("sanction_group_liability_flag", False)),
        sanction_short_tenure_flag=bool(summary.get("sanction_short_tenure_flag", False)),
        sanction_high_interest_flag=bool(summary.get("sanction_high_interest_flag", False)),
        news_sentiment_score=float(summary.get("news_sentiment_score", 0.0)),
        promoter_risk_score=float(summary.get("promoter_risk_score", 0.0)),
        research_litigation_news_count=float(summary.get("research_litigation_news_count", 0.0)),
        research_sector_headwind_score=float(summary.get("research_sector_headwind_score", 0.0)),
        gst_anomaly_score=float(summary.get("gst_anomaly_score", 0.0)),
        bank_anomaly_score=float(summary.get("bank_anomaly_score", 0.0)),
        financials_found_flag=bool(summary.get("financials_found_flag", False)),
        gst_itc_variance_ratio=float(summary.get("gst_itc_variance_ratio", 0.0)),
        gst_itc_top_supplier_share=float(summary.get("gst_itc_top_supplier_share", 0.0)),
        gst_itc_dependency_ratio=float(summary.get("gst_itc_dependency_ratio", 0.0)),
        gst_cash_tax_ratio=float(summary.get("gst_cash_tax_ratio", 0.0)),
        gst_reverse_charge_turnover_ratio=float(summary.get("gst_reverse_charge_turnover_ratio", 0.0)),
        bank_cash_deposit_ratio=float(summary.get("bank_cash_deposit_ratio", 0.0)),
        bank_round_tripping_score=float(summary.get("bank_round_tripping_score", 0.0)),
        bank_top_counterparty_share=float(summary.get("bank_top_counterparty_share", 0.0)),
        bank_counterparty_hhi=float(summary.get("bank_counterparty_hhi", 0.0)),
        bank_total_txn_volume=float(summary.get("bank_total_txn_volume", 0.0)),
        bank_related_party_transfer_share=float(summary.get("bank_related_party_transfer_share", 0.0)),
    )

