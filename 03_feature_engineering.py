"""
================================================================================
03_feature_engineering.py
--------------------------------------------------------------------------------
AI-Powered E-commerce Customer & Sales Analytics Platform
Step 3: Feature Engineering

What this script does:
    1. Loads the cleaned dataset from 01_data_cleaning.py.
    2. Creates date-derived features (year, month, weekday, weekend flag,
       quarter, days-to-ship).
    3. Creates financial features (profit margin %, average item value,
       discount bucket, is-high-value-order flag).
    4. Creates customer-level tenure features (days since signup, order
       recency in days, average basket size).
    5. Encodes a handful of categorical fields for downstream modeling
       (label-encoded copies, kept alongside the original text columns).
    6. Saves the engineered, model-ready dataset to
       output/engineered_ecommerce_data.csv for the segmentation and ML
       scripts to consume.

Run from the /Python folder (after 02_eda.py):
    python 03_feature_engineering.py
================================================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = "./output/cleaned_ecommerce_data.csv"
OUTPUT_PATH = "./output/engineered_ecommerce_data.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, parse_dates=["Order_Date", "Ship_Date", "Customer_Since"])
    print(f"Loaded cleaned dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Order_Year"] = df["Order_Date"].dt.year
    df["Order_Month"] = df["Order_Date"].dt.month
    df["Order_Quarter"] = df["Order_Date"].dt.quarter
    df["Order_Weekday"] = df["Order_Date"].dt.day_name()
    df["Is_Weekend_Order"] = df["Order_Date"].dt.dayofweek.isin([5, 6])
    df["Order_YearMonth"] = df["Order_Date"].dt.strftime("%Y-%m")
    # Days between order placed and shipped (only defined for shipped orders)
    df["Days_To_Ship"] = (df["Ship_Date"] - df["Order_Date"]).dt.days
    print("  Added: Order_Year, Order_Month, Order_Quarter, Order_Weekday, "
          "Is_Weekend_Order, Order_YearMonth, Days_To_Ship")
    return df


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    # Profit margin as a % of sales, guarding against divide-by-zero
    df["Profit_Margin_Pct"] = np.where(df["Sales"] > 0, (df["Profit"] / df["Sales"]) * 100, 0)

    df["Avg_Item_Value"] = df["Sales"] / df["Quantity"].replace(0, np.nan)

    # Bucket discount into readable tiers for BI / segmentation use
    bins = [-0.01, 0, 0.10, 0.20, 1.0]
    labels = ["No Discount", "Low (0-10%)", "Medium (10-20%)", "High (20%+)"]
    df["Discount_Bucket"] = pd.cut(df["Discount"], bins=bins, labels=labels)

    # Flag high-value order lines (top quartile of Sales) - useful for BI filters
    sales_q75 = df["Sales"].quantile(0.75)
    df["Is_High_Value_Order"] = df["Sales"] >= sales_q75

    print("  Added: Profit_Margin_Pct, Avg_Item_Value, Discount_Bucket, Is_High_Value_Order")
    return df


def add_customer_tenure_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    df["Customer_Tenure_Days"] = (reference_date - df["Customer_Since"]).dt.days
    # Recency_Days and Total_Orders / Total_Spend already exist from Part 1's
    # customer rollup - we simply derive a normalized spend-per-order metric.
    df["Avg_Spend_Per_Order"] = np.where(
        df["Total_Orders"] > 0, df["Total_Spend"] / df["Total_Orders"], 0
    )
    print("  Added: Customer_Tenure_Days, Avg_Spend_Per_Order")
    return df


def add_encoded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode key categorical columns into *_Encoded companion columns.
    Original text columns are preserved for readability / BI use; the
    *_Encoded versions are what the ML scripts (05, 06) will consume."""
    cols_to_encode = ["Category", "Sub_Category", "Payment_Method", "Customer_Segment",
                       "Income_Group", "Gender", "State", "Order_Status", "Shipping_Mode"]
    encoders = {}
    for col in cols_to_encode:
        le = LabelEncoder()
        # Shipping_Mode has NaNs for unshipped orders -> encode a placeholder
        series = df[col].fillna("Not_Shipped").astype(str)
        df[f"{col}_Encoded"] = le.fit_transform(series)
        encoders[col] = le
    print(f"  Label-encoded {len(cols_to_encode)} categorical columns "
          f"(new columns suffixed '_Encoded')")
    return df


def main():
    df = load_data()
    reference_date = df["Order_Date"].max()
    print(f"Reference date for tenure/recency calculations: {reference_date.date()}")

    print("\n" + "=" * 78)
    print("DATE FEATURES")
    print("=" * 78)
    df = add_date_features(df)

    print("\n" + "=" * 78)
    print("FINANCIAL FEATURES")
    print("=" * 78)
    df = add_financial_features(df)

    print("\n" + "=" * 78)
    print("CUSTOMER TENURE FEATURES")
    print("=" * 78)
    df = add_customer_tenure_features(df, reference_date)

    print("\n" + "=" * 78)
    print("CATEGORICAL ENCODING")
    print("=" * 78)
    df = add_encoded_columns(df)

    print("\n" + "=" * 78)
    print("SAVE ENGINEERED DATASET")
    print("=" * 78)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved -> {OUTPUT_PATH}")
    print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  New columns added this step: {df.shape[1] - 46}")

    print("\nDone. Next: run 04_customer_segmentation.py")


if __name__ == "__main__":
    main()
