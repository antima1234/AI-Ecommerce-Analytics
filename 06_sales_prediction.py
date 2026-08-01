"""
================================================================================
06_sales_prediction.py
--------------------------------------------------------------------------------
AI-Powered E-commerce Customer & Sales Analytics Platform
Step 6: Machine Learning Model 2 - Sales Prediction

Framing:
    Sales (an order-line's dollar value) is a continuous number, so this is
    a REGRESSION problem, not a classification one. We predict Sales from
    attributes known at the time of purchase (product, quantity, discount,
    category, customer profile) - i.e. features that would be available to
    a system trying to estimate revenue impact before/at checkout.

What this script does:
    1. Loads the engineered dataset from 03_feature_engineering.py.
    2. Builds a feature matrix that explicitly EXCLUDES Profit and any
       post-hoc financial fields (to avoid leaking the target).
    3. Splits train/test, scales numeric features for the linear baseline.
    4. Trains Linear Regression (baseline) and Random Forest Regressor;
       trains XGBoost Regressor too if available.
    5. Evaluates every model with MAE, RMSE, and R-squared; plots
       Actual-vs-Predicted and a feature-importance chart.
    6. Saves the best model to output/models/sales_model.pkl and exports
       predictions to output/sales_predictions.csv.

Run from the /Python folder (after 05_churn_prediction.py):
    python 06_sales_prediction.py
================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

INPUT_PATH = "./output/engineered_ecommerce_data.csv"
MODEL_DIR = "./output/models"
FIG_DIR = "./output/figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42
sns.set_style("whitegrid")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded engineered dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def prepare_features(df: pd.DataFrame):
    """
    Leakage guard: Sales = Quantity * Unit_Price * (1 - Discount), and
    Profit = Sales - Cost - Shipping. We keep Quantity/Unit_Price/Discount
    (they are known BEFORE the sale completes and are legitimately
    predictive/explanatory) but exclude Profit, Shipping_Cost, and any
    other post-transaction financial fields that are only known AFTER
    Sales is already computed.
    """
    df = df.copy()
    cat_cols = ["Category", "Sub_Category", "Payment_Method", "Customer_Segment", "State"]
    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_Enc"] = le.fit_transform(df[col].astype(str))

    feature_cols = [
        "Quantity", "Unit_Price", "Discount",
        "Category_Enc", "Sub_Category_Enc", "Payment_Method_Enc",
        "Customer_Segment_Enc", "State_Enc",
        "Customer_Age", "Order_Month", "Is_Weekend_Order",
    ]
    df["Is_Weekend_Order"] = df["Is_Weekend_Order"].astype(int)

    X = df[feature_cols]
    y = df["Sales"]
    print(f"  Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  Features used: {feature_cols}")
    print("  (Profit/Shipping_Cost excluded - only known AFTER Sales is computed)")
    return X, y, feature_cols


def evaluate(name, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n  [{name}]")
    print(f"    MAE  : {mae:.2f}")
    print(f"    RMSE : {rmse:.2f}")
    print(f"    R2   : {r2:.4f}")
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}


def plot_actual_vs_predicted(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, _, y_pred) in zip(axes, results):
        ax.scatter(y_test, y_pred, alpha=0.3, s=15, color="#1F4E78")
        lims = [0, max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5)
        ax.set_xlabel("Actual Sales ($)")
        ax.set_ylabel("Predicted Sales ($)")
        ax.set_title(f"{name}: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/sales_actual_vs_predicted.png")
    plt.close()
    print("  Saved sales_actual_vs_predicted.png")


def plot_feature_importance(model, feature_cols, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
                palette="crest", legend=False)
    plt.title(f"Feature Importance - {model_name} (Sales Model)", fontsize=13, fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/sales_feature_importance.png")
    plt.close()
    print("  Saved sales_feature_importance.png")


def main():
    df = load_data()
    X, y, feature_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 78)
    print("TRAINING MODELS")
    print("=" * 78)
    results, all_metrics = [], []

    lin = LinearRegression()
    lin.fit(X_train_scaled, y_train)
    pred_lin = lin.predict(X_test_scaled)
    all_metrics.append(evaluate("Linear Regression", y_test, pred_lin))
    results.append(("Linear Regression", lin, pred_lin))

    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    all_metrics.append(evaluate("Random Forest", y_test, pred_rf))
    results.append(("Random Forest", rf, pred_rf))

    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE)
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)
        all_metrics.append(evaluate("XGBoost", y_test, pred_xgb))
        results.append(("XGBoost", xgb, pred_xgb))
    else:
        print("\n  [XGBoost] Not installed in this environment - skipped (optional per spec).")

    print("\n" + "=" * 78)
    print("VISUALIZATIONS")
    print("=" * 78)
    plot_actual_vs_predicted(results, y_test)
    plot_feature_importance(rf, feature_cols, "Random Forest")

    print("\n" + "=" * 78)
    print("MODEL COMPARISON")
    print("=" * 78)
    comparison = pd.DataFrame(all_metrics).set_index("model").round(3)
    print(comparison.to_string())

    best_model_name = comparison["r2"].idxmax()
    best_model = {n: m for n, m, _ in results}[best_model_name]
    print(f"\n  Best model by R2: {best_model_name}")

    print("\n" + "=" * 78)
    print("SAVE MODEL & PREDICTIONS")
    print("=" * 78)
    needs_scaling = best_model_name == "Linear Regression"
    joblib.dump(
        {"model": best_model, "scaler": scaler if needs_scaling else None,
         "feature_cols": feature_cols, "needs_scaling": needs_scaling},
        f"{MODEL_DIR}/sales_model.pkl",
    )
    print(f"  Saved best model ({best_model_name}) -> {MODEL_DIR}/sales_model.pkl")

    X_all_input = scaler.transform(X) if needs_scaling else X
    df["Predicted_Sales"] = best_model.predict(X_all_input)
    df["Prediction_Error"] = df["Sales"] - df["Predicted_Sales"]

    out_cols = ["Order_Item_ID", "Order_ID", "Product_ID", "Category", "Quantity",
                "Unit_Price", "Discount", "Sales", "Predicted_Sales", "Prediction_Error"]
    df[out_cols].to_csv("./output/sales_predictions.csv", index=False)
    print(f"  Saved predictions -> ./output/sales_predictions.csv")

    print("\nAll six Python pipeline steps complete.")


if __name__ == "__main__":
    main()
