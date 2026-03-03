## Intelli-Credit – Everything Implemented Until Now

This file captures **exactly what is implemented so far in this repo**, and what each implemented piece does. It is meant to be an honest status log, not an architecture wish-list.

---

## 1. Repository-Level Status

- **Code implementation status**
  - There is now a **working Python prototype** that:
    - Ingests multiple input formats (structured CSVs and PDFs).
    - Derives heuristic risk signals from each pillar of the problem statement.
    - Runs a transparent rule-based risk engine.
    - Generates a structured CAM document (DOCX).

- **Documentation status**
  - One major architecture document has been created:
    - `INTELLI_CREDIT_ARCHITECTURE.md`

---

## 2. Implemented Artifact: `INTELLI_CREDIT_ARCHITECTURE.md`

**What it is**
- A detailed **architecture specification** for the Intelli-Credit system, aligned with the hackathon problem statement.
- It describes **how every input type** (GST, bank statements, ITRs, PDFs, officer notes, CIBIL, EPFO, related party ledgers, and web research) will flow through:
  - Ingestion and preprocessing.
  - Models and analytics (GNN, anomaly detection, NLP, XGBoost, SHAP).
  - Into the final **Credit Appraisal Memo (CAM)** and **credit decision**.

**How it is organized**
- Sections and their roles:
  - **High-Level System Overview**
    - Defines the end-to-end goal: from multi-source data to CAM + decision + explanations.
    - Maps directly to the three pillars in the hackathon PS:
      - Data Ingestor.
      - Research Agent.
      - Recommendation Engine.
  - **Detailed Data Flow by Input Type**
    - For each input, it follows a strict pattern:
      - **Input** → **Processing / Models** → **Intermediate Outputs** → **Final Use in CAM & Decision**.
    - Covers:
      - **GST Returns (2A/3B)**:
        - Normalization in Databricks, GSTR-2A vs 3B reconciliation.
        - Anomaly detection (Isolation Forest, Autoencoder).
        - Transaction graph + GNN (GraphSAGE/GAT) for circular trading.
      - **Bank Statements**:
        - Parsing & categorization.
        - Cash flow features and anomalies (round tripping, volatility).
        - Extension of the transaction graph.
      - **ITRs & Financials**:
        - Document AI extraction of financial fields.
        - Ratio computation and cross-check vs GST and bank.
      - **Unstructured Documents** (annual reports, minutes, rating reports, shareholding, legal, sanction letters):
        - OCR + LayoutLM/Donut for structure.
        - Risk NLP (FinBERT/DeBERTa) for litigation/governance/rating/covenant risks.
        - Comparative analysis of sanction terms from other banks.
      - **Direct Text Inputs (Officer Notes, Site Visits, Interviews)**:
        - NLP encoding and classification into management/operational risk scores.
        - Logic for using these scores as features / overlays in the final decision.
      - **Base Application Details**:
        - Sector mapping, exposure policy rules, and limit ceilings.
      - **Advanced Credit & Compliance** (CIBIL, EPFO, related party ledger):
        - Extraction of CIBIL and payroll features.
        - Building related party subgraphs for GNN-based risk.
  - **Research Agent (“Digital Credit Manager”)**
    - Specifies:
      - What external sources to hit (MCA, e-Courts, rating sites, news).
      - How NLP is applied for event detection and sentiment.
      - How outputs become litigation and sector risk scores.
  - **Recommendation Engine & Explainability**
    - Defines:
      - XGBoost as the central scoring engine.
      - Feature store concept aggregating all upstream outputs.
      - SHAP-based explanation generation and mapping to human-readable reasons.
  - **CAM Generator**
    - Describes how the Five Cs of Credit structure the CAM:
      - Character, Capacity, Capital, Collateral, Conditions.
    - Explains how model outputs and explanations are inserted into a Word/PDF template.
  - **UI & Orchestration Flow**
    - Outlines the user journey:
      - Upload → Ingest → Analyze → CAM → Download.
  - **Build Checklist**
    - Very explicit checklist of:
      - Architecture items (mostly done).
      - Ingestor tasks (not yet implemented).
      - Research Agent tasks (not yet implemented).
      - Recommendation & CAM generator tasks (not yet implemented).
      - India-specific intelligence tasks (not yet implemented).
  - **Hackathon Requirements Mapping**
    - Shows how the design addresses:
      - Extraction accuracy.
      - Research depth.
      - Explainability.
      - Indian context sensitivity.

**What it does (practically) for the project**
- **Aligns everyone on the target system**:
  - Any teammate can open this file and understand what needs to be built, in what order, and how components interact.
- **Transforms the hackathon PS into a concrete blueprint**:
  - It converts high-level expectations into a set of features, models, and data flows.
- **Acts as an implementation backlog**:
  - The checklist in the file can be used to plan sprints / hackathon tasks and track progress.
- **Justifies design choices to judges**:
  - Clearly documents why specific models (GNN, Isolation Forest, LayoutLM, FinBERT, XGBoost, SHAP) have been selected for each sub-problem.

---

## 3. What Is Explicitly *Not* Implemented Yet

To avoid confusion, these items are **only described in the architecture; they are not coded or wired up** as of now:

- Databricks data ingestion jobs and schemas.
- OCR and Document AI (LayoutLM/Donut) pipelines.
- Transaction graph construction and GNN training/inference.
- Isolation Forest and Autoencoder anomaly detection pipelines.
- FinBERT/DeBERTa-based risk NLP models (training and inference).
- Web/MCA/e-Courts/news crawlers.
- XGBoost training, scoring service, and SHAP-based explanation service.
- CAM template implementation and PDF/DOCX generator.
- Web UI (file upload, notes input, CAM view) and orchestration backend.

These are the **next concrete implementation steps** that will turn the current design into a working prototype.

---

## 4. Implemented Code: Prototype Engine & Multi-Format Ingestion

### 4.1 Core Tech Stack

- **Language**: Python.
- **Key libraries**:
  - `pandas` for CSV ingestion and basic aggregation.
  - `PyPDF2` for simple PDF text extraction.
  - `python-docx` for CAM DOCX generation.
  - `typer` for a clean CLI interface.

### 4.2 Structured Data Ingestion (`src/ingestion.py`)

- **What it does**
  - Provides simple loaders for the main structured file types:
    - `load_gst_returns(path)`: reads a CSV of GST returns.
    - `load_bank_statements(path)`: reads a CSV of bank statements with parsed dates.
    - `load_itr_financials(path)`: reads a CSV of ITR / financial statement metrics.
  - `summarize_inputs(gst_df, bank_df, fin_df, extra_signals=None)`:
    - Computes:
      - From GST: number of periods, total taxable value (if columns exist).
      - From bank: number of months, total inflows, total outflows.
      - From financials: latest year’s revenue, EBITDA, PAT, net worth, total debt.
    - Merges in `extra_signals` coming from other modules (unstructured, qualitative, advanced credit).

- **How it maps to inputs**
  - **GST Returns / Bank Statements / ITRs** from the PS are mapped to simple CSV formats for this prototype.
  - This forms the “structured features” layer for the risk engine and CAM.

---

## 4.2.1 NEW: GSTR-2A vs GSTR-3B Reconciliation (India-Specific) (`src/gst_reconciliation.py` + `src/main.py`)

- **What it adds**
  - The CLI now supports *separate* GST inputs:
    - `--gstr-2a-csv <path>`
    - `--gstr-3b-csv <path>`
  - A production-style reconciliation module that normalizes common export formats and computes **India-specific GST intelligence**:
    - **ITC mismatch**: total 2A vs total 3B variance + variance ratio.
    - **Supplier concentration**: top supplier ITC share and HHI concentration index.
    - **Suspicious vendor heuristics**: count + examples of high-concentration / low-invoice-count suppliers.
    - **Optional supplier clustering** (best-effort; uses sklearn if available).
    - **India-specific ratios from 3B**:
      - ITC dependency ratio.
      - Cash tax ratio.
      - Refund intensity + refund approval ratio.
      - Reverse charge turnover ratio.

- **Where it plugs in**
  - `src/main.py` runs reconciliation when both 2A and 3B are provided and merges features into `extra_signals`.
  - These features are:
    - Logged into the feature store.
    - Passed through to the 5C risk engine (via summary → `RiskInputs`).

- **Why it matters**
  - This closes a major “Indian context sensitivity” gap by making GSTR-2A vs 3B reconciliation **first-class and explainable**, rather than a future architecture note.

### 4.2.2 NEW: Bank Statement Intelligence (`src/bank_intelligence.py` + `src/main.py`)

- **What it adds**
  - A dedicated **bank flows intelligence** layer that computes:
    - `bank_cash_deposit_ratio`: share of cash-like credits over total credits (based on narration keywords).
    - `bank_round_tripping_score`: heuristic score (0–1) for circular money flows using counterparty × month aggregates.
    - `bank_top_counterparty_share` and `bank_counterparty_hhi`: concentration of banking counterparties by transaction volume.
    - `bank_related_party_transfer_share`: best-effort estimate of related-party style transfers based on counterparty strings.
  - These metrics are added to `extra_signals` in `main.py` whenever a bank CSV is provided and are stored in the feature store.

- **How the risk engine uses it**
  - `RiskInputs` now includes the bank intelligence fields.
  - The 5C engine maps them primarily into **Conditions** and **Character**:
    - High `bank_cash_deposit_ratio` → Conditions penalty (possible cash-heavy profile).
    - High `bank_round_tripping_score` → Conditions penalty (round-tripping behaviour suspected).
    - High `bank_top_counterparty_share` → Character penalty (dependence on a narrow counterparty set).
    - High `bank_related_party_transfer_share` → Character penalty (bank flows dominated by inferred related parties).

- **Why it matters**
  - Moves bank statement analysis from simple inflow/outflow totals toward **behavioural cash-flow intelligence**, aligned with Indian circular-trading and related-party risk patterns.

### 4.3 Unstructured Documents Ingestion (`src/unstructured_ingestion.py`)

- **What it does**
  - Accepts multiple PDF paths (annual reports, legal notices, rating reports, sanction letters, etc.).
  - Uses `PyPDF2` to:
    - Extract text page by page and concatenate it.
  - Runs very simple keyword searches to emulate risk NLP:
    - Keywords grouped into buckets like `litigation`, `default`, `pledge`, `downgrade`.
  - Produces:
    - Counts of hits per bucket.
    - A coarse `litigation_risk_score` between 0 and 1 based on frequency of litigation/default terms.

- **How it maps to inputs**
  - Covers **Annual Reports & Financial Statements**, **Board Meeting Minutes**, **Rating Reports**, **Shareholding Patterns**, **Legal Notices**, **Sanction Letters from other banks**.
  - In the full system, this is where LayoutLM/Donut + FinBERT would live; here it’s a fast heuristic stand-in.

### 4.4 Qualitative Notes Processing (`src/qualitative_inputs.py`)

- **What it does**
  - Accepts a list of free-text notes (officer comments, site visit observations, management interview notes).
  - Concatenates and lowercases the text.
  - Counts hits for:
    - **Positive terms** (e.g., “transparent”, “strong management”).
    - **Negative terms** (e.g., “non-cooperative”, “poor controls”).
    - **Low capacity terms** (e.g., “40% capacity”, “underutilized”).
  - Outputs:
    - `management_quality_score` (0–1).
    - `capacity_utilization_penalty` (0–1).
    - Hit counts and text length (for debugging / analysis).

- **How it maps to inputs**
  - Matches the **Direct Text Inputs via UI Panel**:
    - Qualitative notes, site visit observations, management interviews, operational insights.
  - This is the rule-based stand-in for a transformer-based NLP model that would adjust risk based on officer insights.

### 4.5 Advanced Credit & Compliance (`src/advanced_credit.py`)

- **CIBIL Commercial Report (PDF)**
  - `_extract_pdf_text(path)` uses `PyPDF2` to read text from the CIBIL PDF.
  - `analyze_cibil_pdf(path)`:
    - Looks for high-risk terms like “write-off”, “settled”, “wilful defaulter”, and DPD-related phrases.
    - Produces:
      - `cibil_risk_score` (0–1).
      - `cibil_high_risk_hits` and `cibil_dpd_hits`.

- **EPFO / Payroll Statements (CSV)**
  - `analyze_epfo_payroll(csv_path)`:
    - Expects columns like `month`, `employee_id`, `wage`.
    - Computes:
      - `employee_count`.
      - Number of distinct `payroll_months`.
      - `payroll_stability_score` (higher if data covers >= 6 or 12 months).

- **Related Party Transactions Ledger (CSV)**
  - `analyze_related_party_ledger(csv_path)`:
    - Expects `counterparty_name`, `amount`, `type`.
    - Computes:
      - Total transaction volume.
      - Share of the top counterparty.
      - `related_party_risk_score` (higher if one counterparty dominates flows).

- **How it maps to inputs**
  - Directly corresponds to:
    - **CIBIL Commercial Report (PDF)**.
    - **EPFO / Payroll Statement Upload**.
    - **Related Party Transactions Ledger (for Transaction Graph / GNN)**.
  - In the full system, this layer would feed into a richer GNN and credit behavior model; here it provides scalar proxy scores.

### 4.6 Risk Engine (`src/risk_engine.py`)

- **What it does**
  - `RiskInputs` dataclass now contains:
    - Core financials: `latest_revenue`, `latest_ebitda`, `latest_net_worth`, `latest_total_debt`.
    - Bank summary: `bank_total_inflows`, `bank_total_outflows`.
    - Overlay scores:
      - `litigation_risk_score`.
      - `management_quality_score`.
      - `capacity_utilization_penalty`.
      - `cibil_risk_score`.
      - `payroll_stability_score`.
      - `related_party_risk_score`.
  - `simple_rule_based_decision(features, requested_limit, base_rate=10.0)`:
    - Computes a score from:
      - Leverage and revenue vs limit.
      - EBITDA margin.
      - Qualitative management comfort and capacity penalty.
      - CIBIL risk, litigation risk, related party risk.
      - Payroll stability.
    - Normalizes to a 0–1 score.
    - Decides approve/reject with:
      - Scaled `recommended_limit`.
      - `recommended_rate` = base rate + spread (depending on score).
    - Builds a list of detailed **reasons** explaining the decision.
  - `build_risk_inputs_from_summary(summary)`:
    - Converts the combined summary dict (structured + extra signals) into `RiskInputs`.

- **How it maps to PS**
  - This is the prototype for the **Recommendation Engine**.
  - Encodes explainable decision logic similar to what an XGBoost + SHAP setup would learn, but in transparent rules.

### 4.7 CAM Generator (`src/cam_generator.py`)

- **What it does**
  - `generate_cam_docx(output_path, company_name, sector, requested_limit, risk_decision, input_summary)`:
    - Creates a DOCX Credit Appraisal Memo with:
      - **Executive Summary**: decision, requested vs recommended limit, recommended rate, risk score.
      - **Company & Sector** overview.
      - **Five Cs of Credit** structure with:
        - Capacity table (revenue, EBITDA, PAT, net worth, debt).
      - **Model Rationale**: bullet list of reasons from the risk engine.
  - Saves the file under an `output/` directory by default.

- **How it maps to PS**
  - Implements the **CAM Generator** requirement and showcases explainability.

### 4.8 End-to-End CLI (`src/main.py`)

- **What it does**
  - Provides a Typer-based CLI command `run_appraisal` that:
    1. Accepts:
       - **Base Application Details**: `company_name`, `sector`, `requested_limit`.
       - **Structured files**: `--gst-csv`, `--bank-csv`, `--fin-csv`.
       - **Unstructured PDFs**: `--unstructured-pdf` (can be passed multiple times).
       - **Qualitative notes**: `--note` (multiple free-text notes).
       - **Advanced credit data**: `--cibil-pdf`, `--epfo-csv`, `--related-party-csv`.
    2. Ingests all inputs and obtains:
       - Structured summaries.
       - Unstructured risk signals.
       - Qualitative scores.
       - Advanced credit risk scores.
    3. Merges everything into a unified summary and `RiskInputs`.
    4. Runs the rule-based risk engine.
    5. Generates and saves a DOCX CAM.

- **Example usage**
  - Once dependencies are installed:
    ```bash
    python -m src.main run-appraisal \
      --company-name "Demo Co" \
      --sector "Manufacturing" \
      --requested-limit 50000000 \
      --gst-csv data/gst.csv \
      --bank-csv data/bank.csv \
      --fin-csv data/financials.csv \
      --unstructured-pdf data/annual_report.pdf \
      --unstructured-pdf data/legal_notice.pdf \
      --note "Factory found operating at 40% capacity with experienced management." \
      --cibil-pdf data/cibil.pdf \
      --epfo-csv data/epfo.csv \
      --related-party-csv data/related_party.csv
    ```

---

## 5. How to Use This File Going Forward

- After each meaningful implementation step (e.g., “added XGBoost training”, “integrated real GNN transaction graph”), update this file by:
  - Adding a new subsection under a relevant heading.
  - Describing:
    - **What** was implemented.
    - **How** it was implemented (high-level tech stack / approach).
    - **What** it does and **how it plugs into the overall architecture**.
- This keeps a clear, chronological history of progress that you can show to hackathon judges or teammates.

---

## 6. Critical Gaps Checklist (Next Implementation Steps)

Legend:  
- **[ ]** Not started  
- **[~]** Scaffolding / partial implementation  
- **[x]** Implemented in prototype form  

### 6.1 Document AI
- [x] Basic PDF text extraction (PyPDF2).
- [x] Keyword-based risk indicators from unstructured PDFs.
- [~] OCR fallback integration (e.g., Tesseract via `pytesseract`) for scanned PDFs (scaffolded, optional dependency).
- [x] Layout-aware parsing and table extraction (via `pdfplumber` prototype).
- [~] Structured field extraction from financial statements:
  - [x] Revenue (prototype).
  - [x] Debt (prototype).
  - [x] Contingent liabilities (prototype).
  - [x] Auditor remarks / going-concern flags (presence detection).
  - [ ] Notes to accounts (detailed parsing).
- [x] Section segmentation for long annual reports (e.g., MD&A vs financials vs notes – heuristic markers).

### 6.2 Transaction Graph & GNN
- [x] Build transaction graph schemas (nodes = entities, edges = transactions) – prototype using `networkx`.
- [x] Implement graph construction from:
  - [x] GST (supplier/customer flows).
  - [x] Related party ledger.
  - [ ] Bank counterparties (optional).
- [x] Implement cycle detection and basic circular trading indicators (via `simple_cycles`).
- [x] Implement community detection proxy (weakly connected components).
- [x] Add graph-based risk score (combining cycles, centrality, communities).
- [ ] Scaffold GNN model interfaces (GraphSAGE/GAT) for future training.

### 6.3 Anomaly Detection
- [ ] Define time-series features for GST and bank flows (monthly turnover, volatility, seasonality).
- [ ] Implement IsolationForest-based anomaly detection module.
- [ ] Implement simple reconstruction-based anomaly scoring (Autoencoder scaffold).
- [ ] Wire anomaly scores into the risk engine as additional features.

### 6.4 Risk NLP
- [x] Keyword-based heuristic scoring (current implementation).
- [ ] Integrate transformer-based model (FinBERT/DeBERTa) for inference.
- [ ] Design label set and logic for:
  - [ ] Litigation severity.
  - [ ] Going-concern flags.
  - [ ] Auditor qualifications.
  - [ ] Rating downgrade interpretation.
  - [ ] Event extraction (who, what, amount).
- [ ] Wire NLP-derived labels and severities into the risk engine.

### 6.5 Recommendation Engine (ML)
- [x] Rule-based recommendation engine with explicit reasons.
- [ ] Implement XGBoost-based scoring model (training deferred until labeled data is available).
- [ ] Add SHAP-based explainability pipeline.
- [ ] Implement model evaluation (cross-validation, calibration checks).
- [ ] Expose PD / risk band outputs in the CAM and API.

### 6.6 Research Agent
- [ ] Implement MCA data fetch/scraping for company and director information.
- [ ] Implement e-Courts query module for litigation history.
- [ ] Integrate rating agency website parsers (where feasible).
- [ ] Add news crawling / search integration for promoter and sector events.
- [ ] Run Risk NLP on these external sources to generate secondary research scores.

### 6.7 Feature Store
- [~] Design a simple feature registry (feature names, owners, versions) – implicit via JSON schema.
- [x] Implement local/on-disk feature store (JSON per application ID in `data/feature_store/`).
- [x] Add feature logging from all modules into the store (full summary + decision stored).
- [~] Add audit trail support (what features were used for which decision/version) – engine version string recorded.

### 6.8 Scalability & API Layer
- [x] CLI-based end-to-end orchestration.
- [ ] Implement a basic API layer (e.g., FastAPI) exposing an `/appraise` endpoint.
- [ ] Add async orchestration / background jobs for long-running document analysis.
- [ ] Separate modules into services (ingestion, analysis, decision, CAM generation) where practical.
- [ ] (Optional) Add a thin web UI to upload files, enter notes, and download CAM.

---

## 7. New Architectural Pieces Addressing Gaps

### 7.1 Application State Model (`src/application.py`)

- **What it adds**
  - `Application` dataclass with:
    - `id`, `company_name`, `sector`, `requested_limit`.
    - `status` enum (`CREATED` → `INGESTED` → `ANALYZED` → `SCORED` → `CAM_GENERATED`).
    - `features` dict (final feature vector).
    - `decision` dict (approve/reject, limit, rate, score, reasons).
    - `engine_version` string for traceability.
- **How it’s used**
  - CLI now instantiates an `Application` at the start of `run_appraisal` and updates status as the pipeline progresses.

### 7.2 Feature Store (`src/feature_store.py`)

- **What it adds**
  - File-based feature store under `data/feature_store/`:
    - `log_application(app)` writes `app.to_dict()` as `<app_id>.json`.
    - `load_application(app_id)` reads the stored record for reproducibility.
- **Why it matters**
  - Every run now has:
    - A stable `application_id`.
    - Persisted feature vector + decision + engine version.
  - You can now answer: “Can you reproduce this decision?” with the stored JSON.

### 7.3 Policy-Driven Risk Engine (`risk_policy.json` + `src/risk_engine.py`)

- **What it adds**
  - `risk_policy.json` externalizes:
    - Thresholds and weights for:
      - Leverage.
      - Revenue-to-limit.
      - EBITDA margin.
    - Overlay factors for:
      - Management quality.
      - Capacity penalties.
      - CIBIL, litigation, related party, graph, payroll effects.
    - Spread bands for pricing.
  - `risk_engine.py`:
    - Loads `risk_policy.json` at import into `RISK_POLICY`.
    - All major rule weights are pulled from this config instead of hard-coded.
- **Why it matters**
  - Scoring policy can now be tuned per segment/sector **without code changes**.
  - Provides a stepping stone towards sector-specific policies.

### 7.4 Anomaly Detection (`src/anomaly.py`)

- **What it adds**
  - `compute_gst_anomalies(gst_df)`:
    - Aggregates monthly taxable value by `period`.
    - Computes global z-scores vs mean and **rolling-window z-scores**.
    - `gst_anomaly_score` (0–1) based on the maximum of both.
  - `compute_bank_anomalies(bank_df)`:
    - Aggregates monthly net flows.
    - Same pattern: global and rolling z-scores → `bank_anomaly_score`.
- **Integration**
  - `main.py` calls these when GST/bank data is present and merges outputs into `extra_signals`.
  - These are explicitly **statistical anomaly detection prototypes**, not full ML models.

### 7.5 Cross-Source Consistency Checks (`src/ingestion.py`)

- **What it adds**
  - In `summarize_inputs`:
    - Computes:
      - `gst_vs_itr_revenue_ratio` (GST taxable vs latest reported revenue).
      - `bank_inflows_vs_revenue_ratio` (total inflows vs revenue).
- **Why it matters**
  - Begins to capture **consistency risk** across GST, bank, and ITR data.

### 7.7 Risk Bands, PD Proxy & Confidence

- **What it adds**
  - `RiskDecision` now includes:
    - `risk_band` (LOW / MEDIUM / ELEVATED / HIGH).
    - `pd_estimate` (very coarse PD proxy, for demo only).
  - `cam_generator` surfaces:
    - Risk band and PD estimate.
    - Data sources present and a `data_completeness_score`.
- **Why it matters**
  - Provides a clearer **risk banding** and communicates when a decision is based on partial data.

### 7.8 Sector-Specific Policy & Policy Versioning

- **What it adds**
  - `risk_policy.json` now supports a `sector_policies` block with overrides (prototype examples for NBFC and real estate).
  - `risk_engine.get_effective_policy(sector)` merges sector-specific overrides into the base policy.
  - A `POLICY_HASH` is computed from the JSON contents and stored alongside:
    - `policy_hash` and full `policy_snapshot` in the `Application` record.
- **Why it matters**
  - Risk appetite can vary by sector without code changes.
  - Each decision is tied to an exact policy version/snapshot for reproducibility.

### 7.9 Partial Data Logic & Limit Capping

- **What it adds**
  - `RiskInputs` now includes:
    - `data_completeness_score`, `has_gst`, `has_bank`.
  - `simple_rule_based_decision`:
    - Applies score penalties when key sources (GST/bank) are missing or completeness is low.
    - Caps `recommended_limit` when data completeness is below a threshold.
- **Why it matters**
  - Data completeness is now **functional**, not just informational, influencing score and limit.

### 7.10 Graph Visualization Artifact

- **What it adds**
  - `graph_analysis.save_graph_image(G, path)` renders a simple NetworkX graph image.
  - `main.py` saves the graph PNG under `output/graph_<app_id>.png` and passes the path into the summary.
  - `cam_generator` embeds the image into the CAM (with a fallback note if loading fails).
- **Why it matters**
  - Makes the transaction graph pillar **visually inspectable** in the CAM, not just textually described.

### 7.11 Stress Testing / Sensitivity (`src/stress_test.py`)

- **What it adds**
  - `run_stress_tests` runs a few illustrative scenarios:
    - Revenue -20%.
    - EBITDA -30%.
    - Placeholder anomaly stress.
  - CLI flag `--run-stress-tests`:
    - When enabled, stress test results are added under `application.decision["stress_tests"]` in the feature store record.
- **Why it matters**
  - Demonstrates a basic **sensitivity layer** around the rule-based engine, suitable for discussion with judges.


### 7.12 Research Agent & Transformer NLP (Prototype)

- **What it adds**
  - `research_agent.summarize_research(company_name, sector)`:
    - Performs a lightweight news search (via Google News HTML) for the company + sector.
    - Returns headline count and top titles/URLs as external signals.
  - `transformer_nlp.analyze_texts_with_transformer(texts)`:
    - Optionally loads a FinBERT-based sentiment pipeline (if `transformers` is available).
    - Computes positive/negative/neutral share over officer notes + news headlines.
  - `main.py`:
    - Always attempts `summarize_research` for the company.
    - Feeds notes + headlines through the transformer NLP (if available) to derive tone scores.
- **Why it matters**
  - Implements a first version of the **Digital Credit Manager** pillar using live web research and transformer NLP, with graceful degradation if external dependencies are unavailable.

