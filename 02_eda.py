"""
================================================================================
02_eda.py
--------------------------------------------------------------------------------
AI-Powered E-commerce Customer & Sales Analytics Platform
Step 2: Exploratory Data Analysis

What this script does:
    1. Loads the cleaned dataset produced by 01_data_cleaning.py.
    2. Prints descriptive statistics for numeric and categorical fields.
    3. Generates and saves the following charts to /Python/output/figures:
         - correlation_heatmap.png
         - monthly_sales_trend.png
         - top_10_products.png
         - profit_by_category.png
         - sales_by_state.png
         - customer_age_distribution.png
         - discount_vs_profit.png
         - order_status_breakdown.png
    4. Writes a short auto-generated insights file
       (output/eda_insights.txt) summarizing the headline numbers.

Run from the /Python folder (after 01_data_cleaning.py):
    python 02_eda.py
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

INPUT_PATH = "./output/cleaned_ecommerce_data.csv"
FIG_DIR = "./output/figures"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
PALETTE = "crest"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, parse_dates=["Order_Date", "Ship_Date", "Customer_Since"])
    print(f"Loaded cleaned dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def print_descriptive_stats(df: pd.DataFrame):
    print("\n" + "=" * 78)
    print("NUMERIC SUMMARY")
    print("=" * 78)
    print(df[["Quantity", "Unit_Price", "Discount", "Sales", "Profit",
               "Shipping_Cost", "Customer_Age"]].describe().round(2).to_string())

    print("\n" + "=" * 78)
    print("CATEGORICAL BREAKDOWN")
    print("=" * 78)
    for col in ["Category", "Order_Status", "Payment_Method", "Customer_Segment"]:
        print(f"\n-- {col} --")
        print(df[col].value_counts().to_string())


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = ["Quantity", "Unit_Price", "Discount", "Sales", "Shipping_Cost",
                     "Profit", "Customer_Age", "Customer_Lifetime_Value",
                     "Total_Orders", "Recency_Days"]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap - Key Numeric Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png")
    plt.close()
    print("  Saved correlation_heatmap.png")


def plot_monthly_sales_trend(df: pd.DataFrame):
    monthly = df.set_index("Order_Date").resample("ME")[["Sales", "Profit"]].sum()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(monthly.index, monthly["Sales"], marker="o", color="#1F4E78", label="Sales")
    ax1.plot(monthly.index, monthly["Profit"], marker="o", color="#63BE7B", label="Profit")
    ax1.set_title("Monthly Sales & Profit Trend", fontsize=13, fontweight="bold")
    ax1.set_ylabel("USD")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/monthly_sales_trend.png")
    plt.close()
    print("  Saved monthly_sales_trend.png")


def plot_top_products(df: pd.DataFrame):
    top10 = df.groupby("Product_ID")["Sales"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top10.values, y=top10.index, hue=top10.index, palette=PALETTE, legend=False)
    plt.title("Top 10 Products by Total Sales", fontsize=13, fontweight="bold")
    plt.xlabel("Total Sales ($)")
    plt.ylabel("Product ID")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/top_10_products.png")
    plt.close()
    print("  Saved top_10_products.png")


def plot_profit_by_category(df: pd.DataFrame):
    cat_profit = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cat_profit.values, y=cat_profit.index, hue=cat_profit.index,
                palette=PALETTE, legend=False)
    plt.title("Total Profit by Category", fontsize=13, fontweight="bold")
    plt.xlabel("Total Profit ($)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/profit_by_category.png")
    plt.close()
    print("  Saved profit_by_category.png")


def plot_sales_by_state(df: pd.DataFrame):
    state_sales = df.groupby("State")["Sales"].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=state_sales.values, y=state_sales.index, hue=state_sales.index,
                palette=PALETTE, legend=False)
    plt.title("Total Sales by State", fontsize=13, fontweight="bold")
    plt.xlabel("Total Sales ($)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/sales_by_state.png")
    plt.close()
    print("  Saved sales_by_state.png")


def plot_customer_age_distribution(df: pd.DataFrame):
    ages = df.drop_duplicates("Customer_ID")["Customer_Age"]
    plt.figure(figsize=(9, 5))
    sns.histplot(ages, bins=25, kde=True, color="#1F4E78")
    plt.title("Customer Age Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/customer_age_distribution.png")
    plt.close()
    print("  Saved customer_age_distribution.png")


def plot_discount_vs_profit(df: pd.DataFrame):
    sample = df.sample(min(4000, len(df)), random_state=42)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=sample, x="Discount", y="Profit", hue="Category",
                     alpha=0.5, s=25)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Discount % vs Profit (sampled orders)", fontsize=13, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/discount_vs_profit.png")
    plt.close()
    print("  Saved discount_vs_profit.png")


def plot_order_status_breakdown(df: pd.DataFrame):
    status_counts = df.drop_duplicates("Order_ID")["Order_Status"].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(status_counts.values, labels=status_counts.index, autopct="%1.1f%%",
            colors=sns.color_palette(PALETTE, len(status_counts)))
    plt.title("Order Status Breakdown", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/order_status_breakdown.png")
    plt.close()
    print("  Saved order_status_breakdown.png")


def write_insights_file(df: pd.DataFrame):
    monthly = df.set_index("Order_Date").resample("ME")["Sales"].sum()
    best_month = monthly.idxmax().strftime("%B %Y")
    cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    cat_profit = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
    state_sales = df.groupby("State")["Sales"].sum().sort_values(ascending=False)
    return_rate = (df["Return_Flag"] == "Yes").mean()
    corr_disc_profit = df[["Discount", "Profit"]].corr().iloc[0, 1]

    lines = [
        "EDA INSIGHTS SUMMARY (auto-generated from cleaned_ecommerce_data.csv)",
        "=" * 70,
        f"Best-selling category (revenue): {cat_sales.index[0]} (${cat_sales.iloc[0]:,.0f})",
        f"Lowest-selling category (revenue): {cat_sales.index[-1]} (${cat_sales.iloc[-1]:,.0f})",
        f"Most profitable category: {cat_profit.index[0]} (${cat_profit.iloc[0]:,.0f})",
        f"Top state by sales: {state_sales.index[0]} (${state_sales.iloc[0]:,.0f})",
        f"Strongest sales month: {best_month} (${monthly.max():,.0f})",
        f"Overall return rate: {return_rate:.1%}",
        f"Correlation between Discount % and Profit: {corr_disc_profit:.2f} "
        f"({'negative -> higher discounts erode profit' if corr_disc_profit < 0 else 'positive'})",
    ]
    with open("./output/eda_insights.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nSaved output/eda_insights.txt")


def main():
    df = load_data()
    print_descriptive_stats(df)

    print("\n" + "=" * 78)
    print("GENERATING VISUALIZATIONS")
    print("=" * 78)
    plot_correlation_heatmap(df)
    plot_monthly_sales_trend(df)
    plot_top_products(df)
    plot_profit_by_category(df)
    plot_sales_by_state(df)
    plot_customer_age_distribution(df)
    plot_discount_vs_profit(df)
    plot_order_status_breakdown(df)

    print("\n" + "=" * 78)
    print("INSIGHTS SUMMARY")
    print("=" * 78)
    write_insights_file(df)

    print("\nDone. Next: run 03_feature_engineering.py")


if __name__ == "__main__":
    main()
