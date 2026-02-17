# Data Governance Framework: Payment Risk Insight Platform

## Overview
This document defines the data governance framework for the **Payment Risk Insight Platform**. It describes the dataset used, transformation logic, privacy assumptions, quality expectations, and audit-ready logging design that govern all downstream modeling and analysis. 

The goal of this framework is to demonstrate production-grade data handling discipline consistent with real-world payment risk systems.

---

## Data Source
* **Dataset:** PaySim – Synthetic Financial Dataset for Fraud Detection
* **Origin:** Generated using aggregated mobile money transaction logs from a multinational financial services provider
* **Nature:** Fully synthetic; no real customer or merchant data
* **Time Granularity:** Hourly (`step` represents one hour)
* **Time Span:** 744 steps (30 days)

*Note: The dataset is used strictly for research and demonstration purposes.*

---

## Data Lineage & Transformations

### Ingestion
* Raw PaySim CSV files are downloaded manually and stored in `data/raw/`
* Raw data files are never committed to version control
* All transformations are deterministic and reproducible

### Preprocessing Pipeline
The preprocessing pipeline is implemented in `src/data_processing/preprocess_paysim.py` using Polars for performance and scalability. The pipeline performs the following steps:

#### 1. Schema Validation
Required columns are verified before processing. Processing halts if required fields are missing:
* `step`
* `type`
* `amount`
* `nameOrig`
* `nameDest`
* `isFraud`
* `isFlaggedFraud`

#### 2. Explicit Leakage Prevention
The following balance-related fields are intentionally excluded:
* `oldbalanceOrg`
* `newbalanceOrig`
* `oldbalanceDest`
* `newbalanceDest`

*Governance Decision:* These fields are removed due to documented label leakage in the PaySim dataset, as fraudulent transactions are canceled and balances reflect post-hoc outcomes. 

#### 3. Privacy-First Identifier Handling
Account identifiers (`nameOrig`, `nameDest`) are treated as sensitive. 
* Identifiers are hashed using SHA-256 with a static salt.
* Hashing is performed on unique values only for efficiency.
* Raw identifiers are dropped immediately after hashing.
* Resulting fields:
    * `account_id` (hashed origin account)
    * `counterparty_id` (hashed destination account)

*No raw identifiers persist beyond preprocessing.*

#### 4. Time Feature Construction
Time-based features are derived from the `step` field to support time-aware behavioral modeling:
* `hour` = `step % 24`
* `day` = `step // 24`
* `is_night` = indicator for transactions between 00:00–05:00

#### 5. Behavioral Feature Engineering
Behavioral features are computed using native Polars window functions. All rolling computations are ordered by time and grouped by account:
* `txn_count_24h`: rolling count of transactions per account over 24 hours
* `avg_amount_24h`: rolling average transaction amount per account
* `amount_deviation`: deviation from rolling average amount

#### 6. Categorical Encoding
Transaction type (`type`) is renamed to `txn_type`. One-hot encoding is applied to generate:
* `txn_type_CASH_IN`
* `txn_type_CASH_OUT`
* `txn_type_DEBIT`
* `txn_type_PAYMENT`
* `txn_type_TRANSFER`

---

## Output Artifacts
* Processed datasets are written to `data/processed/`
* Example artifact: `paysim_processed_v1.parquet`
* Processed data files are not committed to version control.

---

## Privacy Assumptions
* The dataset is synthetic and contains no real PII.
* All identifiers are treated as sensitive by default.
* Identifiers are hashed prior to modeling and never displayed in dashboards or logs.
* No attempt is made to re-identify entities.

*This approach mirrors privacy-first handling used in production payment systems.*

---

## Data Quality Expectations & Validation
The following quality checks are enforced or documented:

* **Schema Integrity:** Required columns must be present; column types validated during ingestion.
* **Value Constraints:** Transaction amounts must be non-negative; fraud labels must be binary (`0` or `1`).
* **Temporal Consistency:** Time steps must be non-negative; transactions are ordered by time per account for behavioral features.
* **Distribution Sanity:** Fraud rate monitored to ensure consistency with dataset documentation.

*Violations are surfaced prior to model training or inference.*

---

## Audit-Ready Logging Design
Every model inference is designed to generate a structured log entry containing:

* Request ID (UUID)
* Timestamp
* Generated transaction ID
* Model version
* Risk score
* Top contributing features
* Recommended action (Approve / Review / Block)
* AI insight generated (True / False)
* Insight validation passed (True / False)

Logs are machine-readable, deterministic, and suitable for post-hoc analysis and review.

---

## Frozen Feature List (v1)
The following features are frozen for Phase 1 and must not be modified until a new version is explicitly introduced.

**Identifiers**
* `account_id`
* `counterparty_id`

**Transaction Attributes**
* `amount`
* `isFlaggedFraud`

**Time Features**
* `step`
* `hour`
* `day`
* `is_night`

**Behavioral Features**
* `txn_count_24h`
* `avg_amount_24h`
* `amount_deviation`

**Transaction Type Encodings**
* `txn_type_CASH_IN`
* `txn_type_CASH_OUT`
* `txn_type_DEBIT`
* `txn_type_PAYMENT`
* `txn_type_TRANSFER`

**Target Variable**
* `isFraud`

---

## Governance Scope & Limitations
* This project demonstrates compliance-aware design, not regulatory approval.
* Governance controls are illustrative and educational.
* No claims are made regarding production deployment or certification.

## Summary
This governance framework ensures the explicit prevention of label leakage, privacy-first handling of identifiers, deterministic and reproducible transformations, and clear documentation of assumptions and limitations. The design reflects best practices used in real-world payment risk pipelines.