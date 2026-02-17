"""
PaySim Preprocessing Script (Polars Version)
---------------------------
Purpose:
- Prepare PaySim transaction data for fraud risk modeling
- Remove leakage-prone fields
- Hash sensitive identifiers
- Create time-based behavioral features
- Produce a clean, model-ready dataset

Design Principles:
- Privacy-first handling of identifiers
- Explicit prevention of label leakage
- Deterministic, reproducible transformations
"""

import polars as pl
import hashlib
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------

RAW_DATA_PATH = Path("data/raw/transactions.csv")
OUTPUT_PATH = Path("data/processed/paysim_processed_v1.parquet")

HASH_SALT = "paysim_project_salt_v1"  # static salt for reproducibility

# Columns known to leak fraud labels (must be excluded)
LEAKAGE_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

# -----------------------------
# Utility Functions
# -----------------------------

def hash_identifier(value: str) -> str:
    """
    Hash sensitive identifiers using SHA-256 with a static salt.
    """
    if value is None:
        return None
    return hashlib.sha256(f"{value}_{HASH_SALT}".encode()).hexdigest()

def validate_schema(df: pl.DataFrame) -> None:
    """
    Basic schema validation to ensure required columns exist.
    """
    required_columns = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "isFraud",
        "isFlaggedFraud"
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# -----------------------------
# Preprocessing Pipeline
# -----------------------------

def preprocess_paysim(df: pl.DataFrame) -> pl.DataFrame:
    """
    Main preprocessing pipeline for PaySim dataset using Polars.
    """

    # Validate schema
    validate_schema(df)

    # Drop leakage-prone balance columns
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(cols_to_drop)

    # ---------------------------------------------------------
    # OPTIMIZATION 1: Map Unique Hashes instead of Row-by-Row
    # ---------------------------------------------------------
    print("Hashing identifiers...")
    # Extract all unique names from both columns at once
    unique_names = pl.concat([df["nameOrig"], df["nameDest"]]).drop_nulls().unique()
    
    # Process the hashes in pure Python only for unique values
    name_list = unique_names.to_list()
    hash_list = [hash_identifier(n) for n in name_list]
    
    # Create a fast Polars mapping table
    mapping_df = pl.DataFrame({
        "name": name_list,
        "hashed_id": hash_list
    })

    # Join the mapping table back to the main DataFrame for nameOrig
    df = df.join(mapping_df, left_on="nameOrig", right_on="name", how="left")
    df = df.rename({"hashed_id": "account_id"})

    # Join the mapping table back to the main DataFrame for nameDest
    df = df.join(mapping_df, left_on="nameDest", right_on="name", how="left")
    df = df.rename({"hashed_id": "counterparty_id"})

    # Clean up old columns
    df = df.drop(["nameOrig", "nameDest"])

    # Time features
    df = df.with_columns([
        (pl.col("step") % 24).alias("hour"),
        (pl.col("step") // 24).alias("day")
    ])
    
    df = df.with_columns(
        pl.col("hour").is_between(0, 5, closed="both").cast(pl.Int32).alias("is_night")
    )

    # ---------------------------------------------------------
    # OPTIMIZATION 2: Native Window Functions for Behavior
    # ---------------------------------------------------------
    print("Calculating behavioral features...")
    # Sort for time-based features
    df = df.sort(["account_id", "step"])

    # Calculate rolling window functions. Polars 'over' handles the group-by grouping instantly.
    df = df.with_columns([
        # Safely count rows in window by summing 1s for valid steps
        pl.lit(1).rolling_sum(window_size=24, min_periods=1)
            .over("account_id", order_by="step").alias("txn_count_24h"),
        
        # Native rolling mean
        pl.col("amount").rolling_mean(window_size=24, min_periods=1)
            .over("account_id", order_by="step").alias("avg_amount_24h")
    ])

    df = df.with_columns(
        (pl.col("amount") - pl.col("avg_amount_24h")).alias("amount_deviation")
    )

    # Transaction type encoding (One-Hot Encoding)
    # We rename "type" to "txn_type" first so the dummies get generated as "txn_type_CASH_OUT", etc.
    df = df.rename({"type": "txn_type"}).to_dummies("txn_type")

    # Final sanity checks
    # Polars extracts single values via .item()
    assert df.select(pl.col("amount").min()).item() >= 0, "Negative transaction amounts detected"
    assert df.select(pl.col("isFraud").is_in([0, 1]).all()).item(), "Invalid fraud labels detected"

    return df


# -----------------------------
# Execution
# -----------------------------

def main():
    print("Loading raw PaySim data...")
    # Polars loads CSVs significantly faster than Pandas natively
    df_raw = pl.read_csv(RAW_DATA_PATH)

    print("Preprocessing data...")
    df_processed = preprocess_paysim(df_raw)

    print(f"Saving processed dataset to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Polars writes directly to parquet without needing separate Pandas/PyArrow configs
    df_processed.write_parquet(OUTPUT_PATH)

    print("Preprocessing complete.")
    print(f"Final dataset shape: {df_processed.shape}")


if __name__ == "__main__":
    main()