"""
Behavior-Only Feature Engineering (v1.1)
-------------------------------------------------
Adds payments-realistic behavioral signals without using labels or proxies.
Utilizes Polars Lazy API for optimal memory and compute efficiency.
"""

import polars as pl
from pathlib import Path

INPUT_PATH = Path("data/processed/paysim_processed_v1.parquet")
OUTPUT_PATH = Path("data/processed/paysim_features_v1_1.parquet")

def add_features(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    # Ensure sorted for time-diff features
    lazy_df = lazy_df.sort(["account_id", "step"])

    lazy_df = lazy_df.with_columns([
        # 1. Time since last transaction (per account)
        (pl.col("step") - pl.col("step").shift(1).over("account_id")).clip(upper_bound=168)
        .fill_null(0)
        .cast(pl.Int32)
        .alias("time_since_last_txn_hr"),

        # 2. Relative amount vs rolling mean (adding 1e-6 to prevent division by zero)
        (pl.col("amount") / (pl.col("avg_amount_24h") + 1e-6))
        .cast(pl.Float32) 
        .alias("amount_ratio_24h"),

        # 3. High-risk transaction type flags 
        pl.col("txn_type_TRANSFER").fill_null(0).cast(pl.Int8).alias("is_transfer"),
        pl.col("txn_type_CASH_OUT").fill_null(0).cast(pl.Int8).alias("is_cash_out"),
    ])

    # 4. Strict sequence signal: CASH_OUT within 6 time-steps (hours) of a TRANSFER
    lazy_df = lazy_df.with_columns(
        # Find the step number of the last transfer and propagate it forward
        pl.when(pl.col("is_transfer") == 1)
        .then(pl.col("step"))
        .otherwise(None)
        .forward_fill()
        .over("account_id")
        .alias("last_transfer_step")
    ).with_columns(
        # Check if the difference between current step and last transfer step is <= 6
        ((pl.col("step") - pl.col("last_transfer_step")) <= 6)
        .fill_null(False)
        .cast(pl.Int8)
        .alias("transfer_in_last_6h")
    ).with_columns(
        # Final interaction: Is it a cash_out AND was there a transfer in the last 6h?
        (pl.col("transfer_in_last_6h") * pl.col("is_cash_out"))
        .cast(pl.Int8)
        .alias("transfer_then_cashout_6h")
    ).drop("last_transfer_step") # Clean up the intermediate column

    return lazy_df

def main():
    print("Starting Phase 2 Feature Engineering (v1.1)...")
    
    # 1. SCAN instead of READ (Activates Lazy Evaluation)
    q = pl.scan_parquet(INPUT_PATH)
    
    # 2. Build the optimized computation graph
    q_features = add_features(q)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 3. SINK directly to disk (Streams data, hyper-efficient)
    print(f"Executing pipeline and streaming to {OUTPUT_PATH}...")
    q_features.sink_parquet(OUTPUT_PATH)
    
    # Read just the metadata to confirm the final shape
    final_shape = pl.read_parquet(OUTPUT_PATH).shape
    print(f"Feature engineering complete. Final shape: {final_shape}")

if __name__ == "__main__":
    main()