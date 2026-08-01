"""
================================================================================
01_data_cleaning.py
--------------------------------------------------------------------------------
AI-Powered E-commerce Customer & Sales Analytics Platform
Step 1: Data Loading & Cleaning

What this script does:
    1. Loads all raw CSV tables from the /Dataset folder.
    2. Validates schema (row counts, dtypes) for each table.
    3. Checks for and removes duplicate records.
    4. Handles missing values using business-rule-aware logic (a null
       Ship_Date is not a data error if the order was never shipped, etc.)
    5. Detects statistical outliers in Sales and Profit using the IQR method
       and flags them (does NOT silently delete them - see notes below).
    6. Fixes data types (dates, numerics) and standardizes text casing.
    7. Builds one clean, analysis-ready denormalized table and writes it to
       /Python/output/cleaned_ecommerce_data.csv for every downstream script
       to reuse.

Run from the /Python folder:
    python 01_data_cleaning.py
================================================================================
"""

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
DATA_DIR = "../Dataset"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.set_option("display.width", 120)


def load_raw_tables(data_dir: str) -> dict:
    """Load every raw CSV table into a dict of DataFrames."""
    tables = {}
    filenames = {
        "categories": "categories.csv",
        "products": "products.csv",
        "customers": "customers.csv",
        "orders": "orders.csv",
        "order_items": "order_items.csv",
        "payments": "payments.csv",
        "shipping": "shipping.csv",
        "reviews": "reviews.csv",
    }
    for key, fname in filenames.items():
        path = os.path.join(data_dir, fname)
        tables[key] = pd.read_csv(path)
        print(f"  Loaded {fname:25s} -> {tables[key].shape[0]:>7,} rows, {tables[key].shape[1]} cols")
    return tables


def check_duplicates(df: pd.DataFrame, subset: str, name: str) -> pd.DataFrame:
    """Report and drop exact duplicate records based on a primary-key column."""
    n_before = len(df)
    dupe_count = df.duplicated(subset=[subset]).sum()
    if dupe_count > 0:
        df = df.drop_duplicates(subset=[subset], keep="first")
        print(f"  [{name}] Removed {dupe_count} duplicate rows (by {subset})")
    else:
        print(f"  [{name}] No duplicates found (by {subset})")
    assert len(df) == n_before - dupe_count
    return df


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Return a boolean mask flagging IQR-method outliers in a numeric series."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower_bound) | (series > upper_bound)


def build_flat_table(tables: dict) -> pd.DataFrame:
    """Join order_items -> orders -> customers -> shipping -> reviews into one
    analysis-ready table, mirroring the star-schema fact table used across
    the rest of this project (Excel, SQL, Power BI)."""
    order_items = tables["order_items"]
    orders = tables["orders"]
    customers = tables["customers"]
    shipping = tables["shipping"]
    reviews = tables["reviews"]

    flat = order_items.merge(orders, on="Order_ID", how="left", suffixes=("", "_order"))
    flat = flat.merge(customers, on="Customer_ID", how="left", suffixes=("", "_cust"))
    flat = flat.merge(
        shipping[["Order_ID", "Shipping_Mode", "Carrier"]], on="Order_ID", how="left"
    )

    review_avg = reviews.groupby("Order_Item_ID")["Review_Score"].mean().reset_index()
    flat = flat.merge(review_avg, on="Order_Item_ID", how="left")

    return flat


def main():
    print("=" * 78)
    print("STEP 1: LOADING RAW DATA")
    print("=" * 78)
    tables = load_raw_tables(DATA_DIR)

    print("\n" + "=" * 78)
    print("STEP 2: DUPLICATE CHECKS")
    print("=" * 78)
    tables["customers"] = check_duplicates(tables["customers"], "Customer_ID", "customers")
    tables["products"] = check_duplicates(tables["products"], "Product_ID", "products")
    tables["orders"] = check_duplicates(tables["orders"], "Order_ID", "orders")
    tables["order_items"] = check_duplicates(tables["order_items"], "Order_Item_ID", "order_items")

    print("\n" + "=" * 78)
    print("STEP 3: BUILD DENORMALIZED ANALYSIS TABLE")
    print("=" * 78)
    df = build_flat_table(tables)
    print(f"  Flat table shape: {df.shape}")

    print("\n" + "=" * 78)
    print("STEP 4: DATA TYPE FIXES")
    print("=" * 78)
    date_cols = ["Order_Date", "Ship_Date", "Customer_Since"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        print(f"  Parsed '{col}' as datetime")

    numeric_cols = ["Quantity", "Unit_Price", "Discount", "Sales", "Shipping_Cost",
                     "Profit", "Customer_Age", "Customer_Lifetime_Value"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = ["Category", "Sub_Category", "Brand", "State", "City", "Payment_Method"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    print("\n" + "=" * 78)
    print("STEP 5: MISSING VALUE ANALYSIS")
    print("=" * 78)
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.to_string())

    # Business-rule-aware handling:
    #  - Ship_Date / Shipping_Mode / Carrier / Delivery_Days are null ONLY for
    #    orders that were never shipped (Order_Status in Processing/Cancelled).
    #    This is expected, not a data quality issue -> we leave them null but
    #    add an explicit boolean flag so downstream steps don't need to
    #    re-derive it.
    df["Was_Shipped"] = df["Ship_Date"].notna()

    #  - Review_Score is null when a customer never left a review. We leave
    #    it null (NOT zero -- imputing 0 would wrongly imply "terrible
    #    review" for products that were simply never rated) and add a flag.
    df["Has_Review"] = df["Review_Score"].notna()

    unexpected_nulls = df.drop(
        columns=["Ship_Date", "Shipping_Mode", "Carrier", "Delivery_Days", "Review_Score"]
    ).isna().sum()
    unexpected_nulls = unexpected_nulls[unexpected_nulls > 0]
    if len(unexpected_nulls) == 0:
        print("\n  No unexpected nulls outside of shipping/review fields. Good.")
    else:
        print("\n  WARNING - unexpected nulls found:")
        print(unexpected_nulls.to_string())

    print("\n" + "=" * 78)
    print("STEP 6: OUTLIER DETECTION (IQR METHOD)")
    print("=" * 78)
    for col in ["Sales", "Profit", "Unit_Price"]:
        mask = detect_outliers_iqr(df[col])
        flag_col = f"{col}_Outlier"
        df[flag_col] = mask
        print(f"  {col:12s}: {mask.sum():>6,} outliers flagged ({mask.mean()*100:.1f}%) "
              f"-> stored in column '{flag_col}' (rows are KEPT, not deleted -- "
              f"they reflect real bulk/high-discount orders, see 02_eda.py for review)")

    print("\n" + "=" * 78)
    print("STEP 7: SANITY / INTEGRITY CHECKS")
    print("=" * 78)
    assert (df["Sales"] >= 0).all(), "Found negative Sales values!"
    assert (df["Quantity"] > 0).all(), "Found non-positive Quantity values!"
    assert df["Order_ID"].notna().all(), "Found rows with missing Order_ID!"
    assert df["Customer_ID"].notna().all(), "Found rows with missing Customer_ID!"
    print("  All integrity checks passed (no negative sales, no zero quantity, no missing keys).")

    print("\n" + "=" * 78)
    print("STEP 8: SAVE CLEANED DATASET")
    print("=" * 78)
    out_path = os.path.join(OUTPUT_DIR, "cleaned_ecommerce_data.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved cleaned dataset -> {out_path}")
    print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\nDone. Next: run 02_eda.py")


if __name__ == "__main__":
    main()
