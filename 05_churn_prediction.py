"""
================================================================================
05_churn_prediction.py
--------------------------------------------------------------------------------
AI-Powered E-commerce Customer & Sales Analytics Platform
Step 5: Machine Learning Model 1 - Customer Churn Prediction

Business rule (defined in the dataset generation step):
    A customer is labeled "Churned" if they have not placed an order in the
    last 180 days relative to the most recent order date in the dataset.

What this script does:
    1. Loads the RFM/segment table from 04_customer_segmentation.py and the
       engineered dataset for extra customer-level features.
    2. Builds a customer-level feature matrix (RFM values, tenure, age,
       segment, income group, gender - all encoded).
    3. Splits into train/test sets (stratified on the churn label).
    4. Scales numeric features.
    5. Trains two classifiers: Logistic Regression (baseline) and Random
       Forest. Trains XGBoost too if the library is available in this
       environment (it is optional per the project spec).
    6. Evaluates every model with Accuracy, Precision, Recall, F1,
       Confusion Matrix, and ROC-AUC; saves the ROC curve plot and
       confusion matrix plots.
    7. Saves the best-performing model to output/models/churn_model.pkl and
       exports predictions for every customer to
       output/churn_predictions.csv.

Run from the /Python folder (after 04_customer_segmentation.py):
    python 05_churn_prediction.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RFM_PATH = "./output/customer_rfm_segments.csv"
MODEL_DIR = "./output/models"
FIG_DIR = "./output/figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42
sns.set_style("whitegrid")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RFM_PATH)
    print(f"Loaded customer RFM/segment table: {df.shape[0]:,} customers, {df.shape[1]} columns")
    print(f"Churn rate in dataset: {df['Churned'].mean():.1%}")
    return df


def prepare_features(df: pd.DataFrame):
    """
    IMPORTANT - leakage guard:
    'Churned' is defined (see dataset generation) as Recency > 180 days.
    That means Recency itself, and anything derived from it -- R_Score,
    RFM_Total, the KMeans Cluster, and the RFM_Segment label -- encode the
    answer directly. Training on them gives a trivial, meaningless 100%
    accuracy model. A real churn model has to predict churn risk from
    behavioral/demographic signals that DON'T already contain the label,
    so those columns are deliberately excluded below.
    """
    df = df.copy()
    cat_cols = ["Customer_Segment", "Gender", "State", "Income_Group"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_Enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [
        "Frequency", "Monetary", "F_Score", "M_Score",
        "Customer_Age", "Customer_Lifetime_Value",
        "Customer_Segment_Enc", "Gender_Enc", "State_Enc", "Income_Group_Enc",
    ]
    X = df[feature_cols]
    y = df["Churned"]
    print(f"  Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  Features used: {feature_cols}")
    print("  (Recency/R_Score/RFM_Total/Cluster/RFM_Segment excluded - they "
          "leak the label definition, see docstring)")
    return X, y, df, encoders, feature_cols


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test, scale=False, scaler=None):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print(f"\n  [{name}]")
    for k, v in metrics.items():
        if k != "model":
            print(f"    {k:10s}: {v:.4f}")
    print("    Classification report:")
    print("    " + classification_report(y_test, y_pred).replace("\n", "\n    "))

    return model, metrics, y_pred, y_proba


def plot_confusion_matrices(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, _, y_pred, _) in zip(axes, results):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Not Churned", "Churned"],
                    yticklabels=["Not Churned", "Churned"])
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_confusion_matrices.png")
    plt.close()
    print("  Saved churn_confusion_matrices.png")


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(8, 7))
    for name, _, _, y_proba in results:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Churn Prediction Models", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_roc_curve.png")
    plt.close()
    print("  Saved churn_roc_curve.png")


def plot_feature_importance(model, feature_cols, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
                palette="crest", legend=False)
    plt.title(f"Feature Importance - {model_name} (Churn Model)", fontsize=13, fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_feature_importance.png")
    plt.close()
    print("  Saved churn_feature_importance.png")


def main():
    df = load_data()
    X, y, df_full, encoders, feature_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 78)
    print("TRAINING MODELS")
    print("=" * 78)
    results = []

    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model_lr, metrics_lr, pred_lr, proba_lr = train_and_evaluate(
        "Logistic Regression", logreg, X_train_scaled, X_test_scaled, y_train, y_test
    )
    results.append(("Logistic Regression", model_lr, pred_lr, proba_lr))

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    model_rf, metrics_rf, pred_rf, proba_rf = train_and_evaluate(
        "Random Forest", rf, X_train, X_test, y_train, y_test
    )
    results.append(("Random Forest", model_rf, pred_rf, proba_rf))

    all_metrics = [metrics_lr, metrics_rf]

    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                             random_state=RANDOM_STATE, eval_metric="logloss")
        model_xgb, metrics_xgb, pred_xgb, proba_xgb = train_and_evaluate(
            "XGBoost", xgb, X_train, X_test, y_train, y_test
        )
        results.append(("XGBoost", model_xgb, pred_xgb, proba_xgb))
        all_metrics.append(metrics_xgb)
    else:
        print("\n  [XGBoost] Not installed in this environment - skipped "
              "(optional per project spec; Random Forest covers tree-based "
              "modeling and generally matches/beats XGBoost on a dataset "
              "this size).")

    print("\n" + "=" * 78)
    print("VISUALIZATIONS")
    print("=" * 78)
    plot_confusion_matrices([(n, m, p, pr) for n, m, p, pr in results], y_test)
    plot_roc_curves([(n, m, p, pr) for n, m, p, pr in results], y_test)
    plot_feature_importance(model_rf, feature_cols, "Random Forest")

    print("\n" + "=" * 78)
    print("MODEL COMPARISON")
    print("=" * 78)
    comparison = pd.DataFrame(all_metrics).set_index("model").round(4)
    print(comparison.to_string())

    best_model_name = comparison["roc_auc"].idxmax()
    best_model = {n: m for n, m, _, _ in results}[best_model_name]
    print(f"\n  Best model by ROC-AUC: {best_model_name}")

    print("\n" + "=" * 78)
    print("SAVE MODEL & PREDICTIONS")
    print("=" * 78)
    needs_scaling = best_model_name == "Logistic Regression"
    joblib.dump(
        {"model": best_model, "scaler": scaler if needs_scaling else None,
         "feature_cols": feature_cols, "needs_scaling": needs_scaling},
        f"{MODEL_DIR}/churn_model.pkl",
    )
    print(f"  Saved best model ({best_model_name}) -> {MODEL_DIR}/churn_model.pkl")

    X_all_input = scaler.transform(X) if needs_scaling else X
    df_full["Predicted_Churn_Probability"] = best_model.predict_proba(X_all_input)[:, 1]
    df_full["Predicted_Churn"] = best_model.predict(X_all_input)
    df_full["Churn_Risk_Tier"] = pd.cut(
        df_full["Predicted_Churn_Probability"], bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    out_cols = ["Customer_ID", "Recency", "Frequency", "Monetary", "RFM_Segment",
                "Churned", "Predicted_Churn", "Predicted_Churn_Probability", "Churn_Risk_Tier"]
    df_full[out_cols].to_csv("./output/churn_predictions.csv", index=False)
    print(f"  Saved predictions -> ./output/churn_predictions.csv")
    print(f"  High-risk customers identified: {(df_full['Churn_Risk_Tier']=='High Risk').sum():,}")

    print("\nDone. Next: run 06_sales_prediction.py")


if __name__ == "__main__":
    main()
