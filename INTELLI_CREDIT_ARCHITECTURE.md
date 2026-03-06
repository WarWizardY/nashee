## Intelli-Credit – System Architecture & Build Checklist

This document captures the end-to-end architecture for the Intelli-Credit hackathon prototype and a build checklist that tracks where we are vs. the hackathon problem statement.

---

## 1. High-Level System Overview

**Goal**: Given multi-source corporate credit data (structured, unstructured, web-scale research, and officer inputs), automatically produce:
- A structured **Credit Appraisal Memo (CAM)** (Word/PDF) organized around the Five Cs of Credit.
- A **credit decision** (approve/reject), **recommended limit**, and **interest rate/risk premium**.
- **Explainable reasoning**: clear, human-readable reasons and risk factors for the recommendation.

**Core pillars (from PS)**:
- **Pillar 1 – Data Ingestor**: Ingest, clean, and structure heterogeneous data (GST, bank, ITR, PDFs).
- **Pillar 2 – Research Agent (“Digital Credit Manager”)**: Secondary research (web, MCA, courts, news) + primary officer insights input.
- **Pillar 3 – Recommendation Engine**: Score, decide, and generate a CAM with explainable logic.

---

## 2. Detailed Data Flow by Input Type

Each subsection follows the pattern:
**Input → Processing / Models → Intermediate Outputs → Final Use in CAM & Decision**

### 2.1 GST Returns (GSTR-2A & GSTR-3B)

**Input**
- GSTR-2A (supplier-side, invoice-level data).
- GSTR-3B (self-declared summary returns).

**Processing / Models**
- **Ingestion & Normalization (Databricks)**
  - Parse uploaded GST files (JSON/Excel/PDF exports) into bronze tables.
  - Standardize GSTINs, invoice dates, tax amounts, HSN codes to silver tables.
- **Reconciliation & Rule-based Checks**
  - Compare GSTR-2A vs GSTR-3B by period, supplier, and tax amount.
  - Flag under-reporting/over-reporting beyond thresholds (e.g., >10% mismatch).
- **Anomaly Detection (Revenue Inflation / Circular Trading)**
  - Build company ↔ counterparty transaction features (turnover, concentration, growth, seasonality).
  - Train **Isolation Forest** on “normal” historical GST patterns within sector & size buckets.
  - Train an **Autoencoder** (on standardized GST feature vectors) to detect reconstruction error → anomaly score.
- **Transaction Graph Construction (for GNN)**
  - Build a directed weighted graph:
    - **Nodes**: Company, suppliers, customers (by GSTIN / PAN).
    - **Edges**: Taxable value flows over time (weight = amount, features = frequency, seasonality).
  - Use **GraphSAGE/GAT** to learn representations and detect:
    - Circular trading loops.
    - Dense, suspicious clusters.
    - High centrality nodes with abnormal patterns.

**Intermediate Outputs**
- `gst_clean_features` (turnover, growth, top 10 counterparties, mismatch ratios).
- `gst_mismatch_flags` (2A vs 3B discrepancies, fake invoice suspicion).
- `gst_anomaly_score` (Isolation Forest + Autoencoder).
- `gst_circular_trading_risk` (GNN-based risk score / label).

**Final Use in CAM & Decision**
- **CAM Sections**
  - **Capacity / Revenue Quality**: Stability of GST flows, consistency vs sector.
  - **Character & Conditions**: Any indications of tax compliance risk or circular trading.
- **Decision Model Inputs**
  - All above features and risk flags feed into **XGBoost** credit score.
  - High `gst_circular_trading_risk` or extreme `gst_anomaly_score` penalizes score and reduces recommended limit / increases risk premium.

---

### 2.2 Bank Statements

**Input**
- Monthly/quarterly bank statements (CSV/PDF) for company’s operating accounts.

**Processing / Models**
- **Ingestion & Parsing**
  - Parse statements into unified schema (date, amount, counterparty, narrative, balance).
  - Use regex + ML-based classifier to categorize transaction types (salary, GST, vendor, EMI, cash, related party).
- **Cash Flow Feature Engineering**
  - Monthly inflow/outflow, volatility, average balance, minimum balance.
  - DSCR proxies, EMI servicing behavior, cheque bounces, cash intensity metrics.
- **Anomaly Detection**
  - Isolation Forest / Autoencoder on bank flow time-series features to detect:
    - Abnormally spiky inflows.
    - Heavy back-to-back round-tripping.
    - Sudden deterioration in liquidity.
- **Transaction Graph Extension**
  - Augment the GST graph with bank counterparty nodes (by account/IFSC).
  - Capture loops between related parties via bank transfers in addition to GST invoices.

**Intermediate Outputs**
- `bank_cashflow_features` (liquidity, volatility, DSCR proxies, bounce counts).
- `bank_anomaly_score` (flows vs sector and own history).
- `round_tripping_risk` (repeated short-window in-and-out flows).

**Final Use in CAM & Decision**
- **CAM Sections**
  - **Capacity**: Cash flow robustness, ability to service debt.
  - **Character**: Repayment behavior, cheque returns, overdraft discipline.
- **Decision Model Inputs**
  - Features and anomaly flags feed XGBoost; weak bank behavior reduces score and recommended limit.

---

### 2.3 Income Tax Returns (ITRs) & Financials

**Input**
- ITR PDFs / XML, audited financial statements, schedules of P&L and Balance Sheet.

**Processing / Models**
- **Document AI Extraction**
  - Use **OCR** (for scanned PDFs) + **LayoutLM/Donut** to extract:
    - Revenue, EBITDA, PAT, Net Worth, Total Debt, Current Ratio, DSCR, etc.
  - Map extracted values into a standardized financial schema in Databricks.
- **Cross-Validation**
  - Compare declared income/turnover with GST & bank flows.
  - Compute leverage (Debt/Equity), coverage ratios, working capital cycle.

**Intermediate Outputs**
- `financial_ratios` (leverage, coverage, profitability, working capital).
- `income_vs_gst_vs_bank_consistency_score`.

**Final Use in CAM & Decision**
- **CAM Sections**
  - **Capital**: Net worth, leverage, retained earnings trend.
  - **Capacity**: Profit-based debt servicing metrics (DSCR, interest coverage).
- **Decision Model Inputs**
  - Ratios and consistency scores feed into XGBoost and affect limit sizing & pricing.

---

### 2.4 Unstructured Documents (Annual Reports, Minutes, Rating Reports, Shareholding, Legal Notices, Sanction Letters)

**Input**
- Annual report PDFs, board meeting minutes, rating reports, shareholding patterns, legal notices, and sanction letters from other banks.

**Processing / Models**
- **OCR + Document AI**
  - Use OCR for scanned images and pass to **LayoutLM/Donut** or similar models.
  - Extract:
    - Covenants, security details, existing limits and sanction terms.
    - Shareholding pattern, promoter share pledges.
    - Board decisions impacting leverage, expansion, or restructuring.
- **Risk NLP (FinBERT/DeBERTa)**
  - Fine-tune on domain labels:
    - `litigation_risk`, `regulatory_risk`, `governance_risk`, `covenant_breach`, `rating_downgrade`, `going_concern_flag`.
  - Run over text chunks from:
    - Legal notices, auditor notes, board minutes, rating rationales.
- **Sanction Letter Comparison**
  - Extract existing banking limits, pricing, tenor, and security structures from other banks’ sanction letters.
  - Compare requested limit/pricing vs market; derive `sanction_competitiveness_score`.

**Intermediate Outputs**
- `doc_risk_events` (list of detected risk sentences with labels, severity).
- `existing_facilities_summary` (limits, pricing, collateral from other banks).
- `shareholding_risk_flags` (high pledging, concentrated ownership, frequent changes).

**Final Use in CAM & Decision**
- **CAM Sections**
  - **Character**: Governance quality, litigation profile, promoter behavior.
  - **Collateral**: Existing security positions, pari-passu/second charge info from sanction letters.
  - **Conditions**: Key covenants and external constraints.
- **Decision Model Inputs**
  - Convert risk events into numerical features and severity scores for XGBoost.
  - High `litigation_risk` or `covenant_breach` sharply lowers score or triggers auto-reject rules.

---

### 2.5 Direct Text Inputs via UI (Credit Officer Notes, Site Visits, Management Interviews)

**Input**
- Free-text notes entered on the UI:
  - Site visit observations (e.g., “factory running at 40% capacity”).
  - Management interview notes (e.g., succession planning, transparency, strategy).
  - Operational insights (e.g., customer concentration, supply disruptions).

**Processing / Models**
- **NLP Encoding**
  - Encode text using a transformer (e.g., DeBERTa/FinBERT or a sentence embedding model).
- **Risk & Qualitative Signal Extraction**
  - Classify notes into:
    - `operational_strength`, `operational_weakness`, `governance_concern`, `management_quality_high/low`, `capacity_utilization_low`, etc.
  - Generate scalar adjustment factors:
    - `management_quality_score`.
    - `operational_resilience_score`.
    - `capacity_utilization_penalty`.
- **Score Adjustment Logic**
  - These qualitative scores act as:
    - Additional features to XGBoost.
    - Or a post-model overlay adjustment (e.g., ±X bps on pricing, ±Y% on limit within guardrails).

**Intermediate Outputs**
- `qualitative_risk_scores` (management, operations, capacity).
- `qualitative_tags` (human-readable flags for CAM).

**Final Use in CAM & Decision**
- **CAM Sections**
  - **Character**: Management integrity, transparency, responsiveness.
  - **Capacity & Conditions**: Operational bottlenecks, plant utilization, expansion risks.
- **Decision Model Inputs**
  - Adjust the baseline score and pricing based on qualitative risk/comfort.

---

### 2.6 Base Application Details (Company, Sector, Requested Loan Amount)

**Input**
- Company name, sector classification, requested facility type & amount, tenor, purpose.

**Processing / Models**
- Map sector to industry risk buckets and typical leverage norms.
- Use requested amount and tenor to:
  - Check max permissible exposure vs internal policies.
  - Stage constraints in the decision model (e.g., max LTV, max multiple of EBITDA).

**Intermediate Outputs**
- `sector_risk_bucket` (low/medium/high, or 1–5).
- `policy_limit_ceiling` (maximum exposure allowed by policy).

**Final Use in CAM & Decision**
- **CAM Sections**
  - **Conditions**: Sector headwinds/tailwinds, regulatory caps, internal exposure norms.
- **Decision Model Inputs**
  - Sector bucket, requested amount, and policy ceilings constrain or scale the recommendation.

---

### 2.7 Advanced Credit & Compliance Data (CIBIL Commercial, EPFO, Related Party Ledger)

**Input**
- CIBIL commercial report (PDF).
- EPFO / payroll statements.
- Related party transactions ledger.

**Processing / Models**
- **CIBIL Commercial Parsing**
  - OCR + NLP to extract:
    - Historical credit facilities, repayment behavior, write-offs, DPD buckets.
    - CIBIL commercial score and risk class.
  - Convert into structured credit history features.
- **EPFO / Payroll Analysis**
  - Parse contributions by employee, month, and location.
  - Derive:
    - Staff count and trend.
    - Seasonality and volatility (signs of distress or expansion).
    - Compliance behavior (on-time payments).
- **Related Party Ledger → Transaction Graph**
  - Build a dedicated subgraph of related party entities.
  - Add edges for inter-company loans, sales/purchases, guarantees.
  - Feed into GNN for:
    - Round-tripping, complex webs of related entities, and shell-like structures.

**Intermediate Outputs**
- `cibil_features` (score, DPD metrics, write-off indicators).
- `payroll_stability_score`, `employment_trend_features`.
- `related_party_graph_risk` (GNN-derived score).

**Final Use in CAM & Decision**
- **CAM Sections**
  - **Character**: Historical credit discipline from CIBIL.
  - **Capacity**: Payroll stability as a proxy for operational stability.
  - **Capital & Conditions**: Complexity and transparency of group structure.
- **Decision Model Inputs**
  - All of the above become features in XGBoost; high `related_party_graph_risk` discourages large unsecured exposures.

---

## 3. Research Agent (“Digital Credit Manager”) – Web-Scale Secondary Research

**Inputs**
- Company name, promoters’ names, sector, and key IDs (CIN, PAN if available).

**Processing / Models**
- **Web/MCA/e-Courts/iNews Crawlers**
  - Targeted scraping / API calls for:
    - MCA filings (director changes, charges, financials, STRIKE OFF notices).
    - e-Courts data (commercial litigation, insolvency, criminal cases).
    - Rating agency websites and sector research.
    - News portals for promoter/company/sector mentions.
- **NLP Risk & Sentiment Analysis**
  - Use FinBERT/DeBERTa models for:
    - Document-level sentiment and event detection (default, fraud, regulatory penalty, rating action).
    - Entity-level aggregation (per promoter / company).

**Outputs**
- `secondary_litigation_risk_score`.
- `secondary_news_sentiment_score` (recent negative/positive news).
- `sector_headwinds_flags` (e.g., RBI circulars, NBFC guidelines).

**Use in CAM & Decision**
- Enrich **Character** and **Conditions** with external, up-to-date intelligence.
- Feed risk scores to XGBoost and CAM narrative (e.g., “High litigation risk due to multiple pending e-Courts cases”).

---

## 4. Recommendation Engine & Explainability

### 4.1 Feature Store & Scoring Model (XGBoost)

**Inputs**
- All structured features and risk scores from:
  - GST, Bank, ITR/Financials.
  - Document risk NLP, secondary research.
  - CIBIL, EPFO, related party GNN, and qualitative officer notes.

**Processing**
- Consolidate into a **feature store** keyed by application ID.
- Train an **XGBoost** model to output:
  - **PD-like risk score** or credit rating band.
  - **Recommended limit** (or scaling of requested limit).
  - **Risk premium / spread** over base rate.

**Outputs**
- `credit_score` (0–1 or rating band).
- `recommended_limit`.
- `recommended_rate` / `risk_spread`.

### 4.2 Explainability (SHAP + Natural Language)

**Processing**
- Use **SHAP** to compute:
  - Top positive and negative feature contributions.
- Map important SHAP features into human-readable explanation templates:
  - Example:
    - “Rejected due to **high litigation risk** from e-Courts data and **abnormally high circular trading indicators in GST graph**, despite **strong GST turnover**.”

**Outputs**
- `explanation_items` (list of bullet points with reason and feature).
- `decision_rationale` (short paragraph for CAM).

---

## 5. CAM Generator (Word/PDF)

**Inputs**
- All feature summaries, risk scores, SHAP explanations, and qualitative tags.

**Processing**
- Fill a pre-defined **CAM template** with:
  - **Executive Summary**: decision (approve/reject), limit, pricing, tenure.
  - **Five Cs of Credit**:
    - **Character**: CIBIL, litigation, governance, management quality, news.
    - **Capacity**: GST & bank flows, DSCR, cash flows.
    - **Capital**: net worth, leverage, retained earnings.
    - **Collateral**: security offered, external sanctions, charges (from MCA/sanction letters).
    - **Conditions**: sector/regulatory context, covenants, exposure norms.
  - **Risk Summary and Mitigants**.
  - **Model Explanation** (SHAP-derived narrative).
- Export as **Word (DOCX)** and/or **PDF** for download.

**Outputs**
- `CAM_document.docx` / `CAM_document.pdf`.

---

## 6. UI & Orchestration Flow

**Steps**
1. Credit Officer uploads structured files (GST, bank, ITR) and PDFs (CIBIL, annual reports, etc.).
2. Officer enters base application details and qualitative notes.
3. System triggers:
   - Databricks ingestion jobs.
   - Document AI extraction.
   - Research Agent crawls for external information.
4. Models (GNN, anomalies, NLP, XGBoost) run and store outputs.
5. CAM generator produces draft memo and explanation.
6. Officer reviews CAM, can add comments, and downloads/export.

---

## 7. Build Checklist (Status)

Legend:  
- **[ ]** Not started  
- **[~]** In progress  
- **[x]** Completed (architecture/design-level only unless marked “implemented”)  

### 7.1 Architecture & Design
- [x] Overall Intelli-Credit conceptual architecture.
- [x] Input-to-output data flow for all major input types.
- [ ] Finalize component diagram and APIs between services.

### 7.2 Data Ingestor (Pillar 1)
- [x] Identify required structured data: GST (2A/3B), bank, ITR.
- [x] Identify required unstructured documents: annual reports, minutes, rating, shareholding, legal, sanction letters.
- [ ] Implement Databricks ingestion pipelines for GST, bank, ITR.
- [ ] Define schemas for GST, bank, and financials (bronze/silver tables).
- [x] Implement OCR + Document AI pipeline for PDFs (LayoutLM/Table Transformer integration).
- [ ] Implement GST 2A vs 3B reconciliation logic and mismatch flags.
- [ ] Implement anomaly detection models (Isolation Forest + Autoencoder) for GST/bank.
- [x] Implement transaction graph building and GNN (GraphSAGE/GAT) training/inference.

### 7.3 Research Agent (Pillar 2)
- [x] Define scope of secondary research (MCA, e-Courts, news, rating, sector reports).
- [ ] Implement crawlers / connectors for MCA, e-Courts, and news sources.
- [ ] Implement FinBERT/DeBERTa-based risk & sentiment classification for external text.
- [ ] Design and implement UI input panel for officer qualitative notes.
- [ ] Implement NLP pipeline to convert officer notes into risk/comfort scores.

### 7.4 Recommendation Engine (Pillar 3)
- [x] Choose XGBoost + SHAP as core decision & explainability stack.
- [ ] Define full feature store schema (all features from GST, bank, ITR, documents, research, and qualitative inputs).
- [ ] Implement XGBoost training and scoring service.
- [ ] Implement SHAP explanation generation and mapping to human-readable narratives.

### 7.5 CAM Generator
- [x] Define CAM structure around Five Cs of Credit.
- [ ] Design CAM template (Word / DOCX) with placeholders.
- [ ] Implement CAM population service (fill template from model outputs).
- [ ] Implement PDF/DOCX export and download.

### 7.6 UI & Orchestration
- [x] Define end-to-end user journey (upload → analyze → CAM).
- [ ] Implement web UI (upload forms, status, CAM viewer).
- [ ] Implement backend orchestration (trigger pipelines, track application ID, manage async processing).

### 7.7 India-Specific Intelligence & Compliance
- [x] Capture India-specific requirements (GSTR-2A vs 3B, CIBIL, EPFO, MCA, RBI/NBFC context).
- [ ] Implement CIBIL commercial report extraction and feature generation.
- [ ] Implement EPFO/payroll parsing and stability scoring.
- [ ] Implement sector risk bucket logic (regulatory and cyclical sectors).

---

## 8. How This Meets Hackathon Requirements

- **Extraction Accuracy**:  
  - OCR + Document AI (LayoutLM/Donut) + specialized schemas target messy Indian PDFs (annual reports, CIBIL, sanction letters).
- **Research Depth**:  
  - Research Agent hits MCA, e-Courts, rating sites, and news for promoter and sector risks.
- **Explainability**:  
  - XGBoost + SHAP + templated natural language explanations provide transparent decision logic.
- **Indian Context Sensitivity**:  
  - Explicit handling of GSTR-2A vs 3B, CIBIL commercial, EPFO, and sector-specific RBI/NBFC regulations feed into the score and CAM narrative.

