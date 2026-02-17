"""
Phase 2: XGBoost Baseline Model (v1)
------------------------------------
Trains a baseline XGBoost classifier using a time-aware split.
Implements early stopping, class imbalance scaling, and 
strict exclusion of leaky/identifier columns.
"""

import json
import polars as pl
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path("data/processed/paysim_features_v1_1.parquet")
MODEL_PATH = Path("artifacts/models/xgb_v1.joblib")
METRICS_PATH = Path("artifacts/metrics/xgb_v1_metrics.json")

# Explicit Split Boundaries
TRAIN_END_DAY = 20
VAL_START_DAY = 21
VAL_END_DAY = 25
TEST_START_DAY = 26

TARGET = "isFraud"
# Drop IDs, Target, and leaky time-markers (model should learn behavior, not calendar dates)
# isFlaggedFraud is also dropped as it is a legacy heuristic rule in PaySim
DROP_COLS = [TARGET, "account_id", "counterparty_id", "step", "day", "isFlaggedFraud"]

# -----------------------------
# Pipeline Functions
# -----------------------------
def time_split(df: pl.DataFrame):
    """Splits data chronologically to mimic production deployment."""
    train = df.filter(pl.col("day") <= TRAIN_END_DAY)
    val = df.filter((pl.col("day") >= VAL_START_DAY) & (pl.col("day") <= VAL_END_DAY))
    test = df.filter(pl.col("day") >= TEST_START_DAY)
    return train, val, test

def to_xy(df: pl.DataFrame, feature_cols: list):
    """Converts Polars DataFrame directly to Numpy for memory efficiency."""
    X = df.select(feature_cols).to_numpy()
    y = df.select(TARGET).to_numpy().ravel()
    return X, y

def print_split_sanity(name: str, y):
    """Prints sanity-check metrics for each split."""
    total = len(y)
    frauds = y.sum()
    rate = (frauds / total) * 100 if total > 0 else 0
    print(f"{name} Set: {total:,} rows | {frauds:,} frauds | {rate:.3f}% fraud rate")

# -----------------------------
# Execution
# -----------------------------
def main():
    print("Loading data...")
    df = pl.read_parquet(DATA_PATH)
    
    # Identify final feature columns dynamically
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    print(f"Training on {len(feature_cols)} features: {feature_cols}\n")

    # 1. Split Data
    train_df, val_df, test_df = time_split(df)

    # 2. Convert to Numpy arrays 
    X_train, y_train = to_xy(train_df, feature_cols)
    X_val, y_val = to_xy(val_df, feature_cols)
    X_test, y_test = to_xy(test_df, feature_cols)

    # Sanity Checks
    print_split_sanity("Train", y_train)
    print_split_sanity("Validation", y_val)
    print_split_sanity("Test", y_test)
    print("-" * 30)

    # 3. Handle class imbalance
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = 50 #neg / max(pos, 1)

    # 4. Initialize Model
    # Note: early_stopping_rounds is added to the constructor in modern XGBoost
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20, 
        random_state=42,
        n_jobs=-1,
    )

    # 5. Train Model
    print("\nTraining XGBoost (with early stopping on Validation set)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50 # Print progress every 50 trees
    )

    # 6. Evaluate
    print("\nEvaluating Model...")
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        "val_auc_roc": float(roc_auc_score(y_val, p_val)),
        "val_auc_pr": float(average_precision_score(y_val, p_val)),
        "test_auc_roc": float(roc_auc_score(y_test, p_test)),
        "test_auc_pr": float(average_precision_score(y_test, p_test)),
        "best_iteration": int(model.best_iteration),
        "scale_pos_weight": float(scale_pos_weight),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "feature_count": int(len(feature_cols)),
    }

    # 7. Save Artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "features": feature_cols}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("\nFinal Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()