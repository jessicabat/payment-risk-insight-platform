# Payment Risk Insight Platform
A production-style payment fraud risk system that combines behavior-based machine learning, economic decision optimization, explainable AI, and a governed GenAI explanation layer—wrapped in an analyst-facing interface.

## Overview
This project simulates a real-world payment risk platform designed to:

- Detect high-risk transactions using behavioral signals.
- Optimize fraud decisions based on economic tradeoffs (profit vs. friction).
- Provide transparent, auditable explanations for every decision using SHAP.
- Support human analysts with a safe, governed GenAI explanation layer.

Rather than focusing on model experimentation alone, the system is built end-to-end as a **decisioning product**, from data governance through to the analyst experience.

[🌐 Project Website](https://jessicabat.github.io/payment-risk-insight-platform/)

## System Performance & Business Metrics
The system was evaluated on a held-out chronological test set to simulate production deployment.

### Model & Economic Decisioning

- **Test Set Volume:** 89,698 transactions
- **Optimal Policy Threshold:** 0.09 *(Economically optimized, sweeping from 0.01 to 0.99)*
- **Fraud Capture (Recall):** 81.46%
- **Approval Rate:** 89.79%
- **Net Value Generated:** 131M local currency units (at optimal threshold)

### Explainability & GenAI Telemetry

- **Explainability Coverage:** 100% of declined transactions include SHAP drivers and raw feature values.
- **GenAI Engine:** Local, air-gapped Llama 3.1 (8B) via Ollama.
- **Guardrail Compliance:** 100% pass rate on strictly defined exclusion terms (e.g., preventing hallucinations of geography or fiat currency).
- **Auditability:** Every generation logs latency (ms), guardrail status, model version, and policy version to an append-only `.jsonl` file.

## System Architecture
```
Raw Data → Feature Engineering → Risk Model → Economic Policy
              ↳ Explainability (SHAP) → GenAI Narratives
              ↳ Analyst Dashboard + Audit Logging
```

### 1. Data & Feature Engineering
- **Dataset:** PaySim (simulated mobile money transactions). All monetary values are expressed in unspecified local currency units to prevent geographic assumptions.
- **Feature Strategy:** Strictly behavior-only features to prevent temporal or identifier leakage. Features include transaction velocity, recency (time since last transaction), amount deviation from recent baselines, and sequence risk (e.g., `TRANSFER` followed immediately by `CASH_OUT`).

### 2. Risk Model & Evaluation
**Model:** XGBoost (gradient-boosted decision trees). Chosen for its robust handling of extreme class imbalance and native compatibility with SHAP.
**Evaluation:** Train/validation/test splits are strictly chronological (by day) to prevent future-data leakage and accurately simulate deployment conditions.

### 3. Economic Tradeoff Analysis
Rather than using a default `0.5` probability cutoff, the system selects a threshold that maximizes net economic value.

- **Revenue Kept:** Approved legitimate transactions × margin rate.
- **Fraud Loss:** 100% loss on missed fraudulent transactions.
- **Friction Cost:** Fixed operational penalty per False Positive (good user blocked).

The threshold that maximizes net profit (`0.09`) is frozen as a versioned JSON Policy Artifact.

### 4. Explainability & GenAI Layer
- **SHAP:** Computes global feature importance for governance and local explanations for individual transactions.
- **GenAI (Llama 3.1):** Translates the mathematical SHAP arrays into human-readable narratives for risk analysts.
- **Governance Principle:** GenAI never makes decisions. It only explains the decisions made upstream by the frozen economic policy. Strict guardrails prevent the LLM from hallucinating data outside the provided payload.

### 5. Analyst Dashboard
An interactive Streamlit application serving as the primary UI for Risk Operations. It features:
- A queue of high-risk alerts.
- Visual indicators of the Risk Score vs. the Economic Threshold.
- A horizontal bar chart of the primary SHAP behavioral drivers.
- On-demand, localized GenAI narrative generation with inline latency and telemetry reporting.

## Assumptions & Scope

- All monetary values are expressed in unspecified local currency units provided by the PaySim dataset.
- No geographic, exchange rate, or purchasing power assumptions are made.
- The GenAI layer is used strictly for explanation and has no decision authority.
- This project simulates a production‑style system but is not intended for live deployment.


## Repository Structure
```
src/
├── app/                  # Streamlit dashboard and CLI interfaces
├── audit/                # Telemetry and JSONL logging infrastructure
├── data_processing/      # Polars-optimized ETL and feature engineering
├── genai/                # Local LLM integration and strict guardrails
└── models/               # XGBoost training and SHAP explainability
notebooks/                # Economic threshold optimization analysis
artifacts/                # Frozen policies, models, and SHAP payloads
docs/                     # Website assets
```

## Demo
A video walkthrough of the Analyst Dashboard and GenAI integration can be viewed on my [🌐 Project Website](https://jessicabat.github.io/payment-risk-insight-platform/)

## Running the Project Locally

This project is designed to run locally for exploration and demonstration purposes.

### Prerequisites
- Python 3.10+
- Ollama (for local LLM inference)
- Git

### Setup

```bash
git clone https://github.com/jessicabat/payment-risk-insight-platform.git
cd payment-risk-insight-platform

# Start the GenAI Engine
ollama pull llama3.1:8b
ollama serve

# Launch the Analyst Dashboard
streamlit run src/app/dashboard.py
```
The dashboard will load precomputed models, policies, and explainability artifacts from the `artifacts/` directory.
