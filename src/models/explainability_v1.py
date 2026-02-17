"""
Phase 2: SHAP Explainability (v1)
---------------------------------
Generates global feature importance and highly detailed, 
context-aware local explanations for the highest-risk transactions.
Optimized with Polars Lazy API.
"""

import json
import joblib
import numpy as np
import polars as pl
import shap
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path("data/processed/paysim_features_v1_1.parquet")
MODEL_PATH = Path("artifacts/models/xgb_v1.joblib")

OUT_DIR = Path("artifacts/explainability")
GLOBAL_OUT = OUT_DIR / "xgb_v1_global_importance.json"
LOCAL_OUT = OUT_DIR / "xgb_v1_sample_explanations.json"

TARGET = "isFraud"
TEST_START_DAY = 26
SHAP_SAMPLE_SIZE = 2000  # For global baseline
TOP_RISK_SAMPLES = 200   # How many high-risk transactions to save for Analyst/LLM review
TOP_K_FEATURES = 5       # Top drivers per transaction

# -----------------------------
# Execution
# -----------------------------
def main():
    print("Loading model...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["features"]

    print("Lazy loading and filtering test data...")
    # OPTIMIZATION: Use scan_parquet to filter out 90% of the data before loading to RAM
    df_test = (
        pl.scan_parquet(DATA_PATH)
        .filter(pl.col("day") >= TEST_START_DAY)
        .collect()
    )

    X_test = df_test.select(feature_cols).to_numpy()
    y_test = df_test.select(TARGET).to_numpy().ravel()

    print("Scoring test data to identify high-risk transactions...")
    # Get actual risk probabilities 
    risk_scores = model.predict_proba(X_test)[:, 1]

    # Combine highest risk transactions with a random sample for a balanced global view
    # np.argsort returns indices from lowest to highest, so we slice from the end [::-1]
    top_risk_indices = np.argsort(risk_scores)[::-1][:TOP_RISK_SAMPLES]
    
    np.random.seed(42)
    random_indices = np.random.choice(len(X_test), size=SHAP_SAMPLE_SIZE, replace=False)
    
    # Unique indices to compute SHAP on (riskiest + random baseline)
    shap_indices = np.unique(np.concatenate([top_risk_indices, random_indices]))
    X_sample = X_test[shap_indices]

    print(f"Computing SHAP values for {len(X_sample)} targeted samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # -----------------------------
    # Global Importance
    # -----------------------------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = sorted(
        [
            {"feature": f, "mean_abs_shap": float(v)}
            for f, v in zip(feature_cols, mean_abs_shap)
        ],
        key=lambda x: x["mean_abs_shap"],
        reverse=True,
    )

    # -----------------------------
    # Local Explanations (Targeting Top Risk)
    # -----------------------------
    local_explanations = []

    print(f"Extracting context for the Top {TOP_RISK_SAMPLES} highest-risk transactions...")
    for idx in top_risk_indices:
        # Find where this index lives in our localized SHAP array
        shap_idx = np.where(shap_indices == idx)[0][0]
        
        # Get exact feature values for this specific transaction
        raw_feature_values = X_test[idx]
        
        # Zip features, SHAP impacts, and actual raw values together
        contribs = list(zip(feature_cols, shap_values[shap_idx], raw_feature_values))
        
        # Sort by absolute SHAP impact to find the top drivers
        top_drivers = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)[:TOP_K_FEATURES]

        local_explanations.append({
            "transaction_index": int(idx),
            "predicted_risk_score": float(risk_scores[idx]),
            "actual_is_fraud_label": int(y_test[idx]),
            "top_risk_drivers": [
                {
                    "feature": f,
                    "shap_impact": float(s),
                    "actual_value": float(v) # Crucial for GenAI context
                }
                for f, s, v in top_drivers
            ],
        })

    # -----------------------------
    # Save Artifacts
    # -----------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_OUT.write_text(json.dumps(global_importance[:20], indent=2))
    LOCAL_OUT.write_text(json.dumps(local_explanations, indent=2))

    print(f"\nSuccess! Artifacts saved to {OUT_DIR}")
    
    # Sanity check printout for the absolute riskiest transaction
    riskiest = local_explanations[0]
    print(f"\n[Preview] Riskiest Transaction (Score: {riskiest['predicted_risk_score']:.4f})")
    print(f"Actual Fraud Label: {riskiest['actual_is_fraud_label']}")
    for driver in riskiest['top_risk_drivers']:
        print(f" -> {driver['feature']}: SHAP {driver['shap_impact']:.2f} (Value: {driver['actual_value']:.2f})")

if __name__ == "__main__":
    main()