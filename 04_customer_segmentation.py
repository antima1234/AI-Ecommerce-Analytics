"""
================================================================================
04_customer_segmentation.py
--------------------------------------------------------------------------------
AI-Powered E-commerce Customer & Sales Analytics Platform
Step 4: RFM Analysis & Customer Segmentation

What this script does:
    1. Loads the engineered dataset from 03_feature_engineering.py.
    2. Aggregates order-item-level data up to one row per customer.
    3. Computes classic RFM metrics:
           Recency   = days since the customer's most recent order
           Frequency = number of distinct orders placed
           Monetary  = total sales generated
    4. Scores each metric into quintiles (1-5) and combines them into an
       RFM segment label (e.g. "Champions", "At Risk", "Lost").
    5. Runs a KMeans clustering model on the scaled RFM values as a second,
       data-driven segmentation (cluster labels 0-3), and profiles each
       cluster.
    6. Saves:
           output/customer_rfm_segments.csv  (one row per customer)
           output/figures/rfm_segment_distribution.png
           output/figures/kmeans_cluster_scatter.png
           output/figures/elbow_method.png

Run from the /Python folder (after 03_feature_engineering.py):
    python 04_customer_segmentation.py
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

INPUT_PATH = "./output/engineered_ecommerce_data.csv"
OUTPUT_PATH = "./output/customer_rfm_segments.csv"
FIG_DIR = "./output/figures"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style("whitegrid")
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, parse_dates=["Order_Date"])
    print(f"Loaded engineered dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def build_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    reference_date = df["Order_Date"].max()

    rfm = df.groupby("Customer_ID").agg(
        Recency=("Order_Date", lambda x: (reference_date - x.max()).days),
        Frequency=("Order_ID", "nunique"),
        Monetary=("Sales", "sum"),
    ).reset_index()

    # Bring along static customer attributes for profiling/BI
    cust_attrs = df.drop_duplicates("Customer_ID")[
        ["Customer_ID", "Customer_Segment", "Customer_Age", "Gender", "State",
         "Income_Group", "Customer_Lifetime_Value", "Churned"]
    ]
    rfm = rfm.merge(cust_attrs, on="Customer_ID", how="left")
    print(f"  Built RFM table for {len(rfm):,} customers")
    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Quintile-score each RFM dimension (5 = best). Recency is scored in
    reverse since a LOWER recency (more recent order) is better."""
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5,
                              labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def label_segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New Customers"
        elif r <= 2 and f >= 4:
            return "At Risk"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Lost"
        else:
            return "Needs Attention"

    rfm["RFM_Segment"] = rfm.apply(label_segment, axis=1)
    print("  Segment distribution:")
    print(rfm["RFM_Segment"].value_counts().to_string())
    return rfm


def run_kmeans_clustering(rfm: pd.DataFrame) -> pd.DataFrame:
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Elbow method to justify k=4
    inertias = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(scaled)
        inertias.append(km.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o", color="#1F4E78")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/elbow_method.png")
    plt.close()
    print("  Saved elbow_method.png")

    k_final = 4
    kmeans = KMeans(n_clusters=k_final, random_state=RANDOM_STATE, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    cluster_profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(1)
    print(f"\n  KMeans cluster profile (k={k_final}):")
    print(cluster_profile.to_string())

    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="Cluster",
                     palette="crest", alpha=0.6, s=30)
    plt.title("Customer Clusters: Frequency vs Monetary Value", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/kmeans_cluster_scatter.png")
    plt.close()
    print("  Saved kmeans_cluster_scatter.png")

    return rfm


def plot_segment_distribution(rfm: pd.DataFrame):
    counts = rfm["RFM_Segment"].value_counts()
    plt.figure(figsize=(9, 6))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, palette="crest", legend=False)
    plt.title("Customer Count by RFM Segment", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rfm_segment_distribution.png")
    plt.close()
    print("  Saved rfm_segment_distribution.png")


def main():
    df = load_data()

    print("\n" + "=" * 78)
    print("BUILDING RFM TABLE")
    print("=" * 78)
    rfm = build_rfm_table(df)

    print("\n" + "=" * 78)
    print("SCORING RFM & LABELING SEGMENTS")
    print("=" * 78)
    rfm = score_rfm(rfm)

    print("\n" + "=" * 78)
    print("KMEANS CLUSTERING (DATA-DRIVEN SEGMENTATION)")
    print("=" * 78)
    rfm = run_kmeans_clustering(rfm)

    print("\n" + "=" * 78)
    print("VISUALIZATIONS")
    print("=" * 78)
    plot_segment_distribution(rfm)

    print("\n" + "=" * 78)
    print("SAVE RESULTS")
    print("=" * 78)
    rfm.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved -> {OUTPUT_PATH}")
    print(f"  {len(rfm):,} customers scored and segmented")

    print("\nDone. Next: run 05_churn_prediction.py")


if __name__ == "__main__":
    main()
