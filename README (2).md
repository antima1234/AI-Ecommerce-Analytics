# AI-Powered E-commerce Customer & Sales Analytics Platform

An end-to-end data analytics portfolio project covering data engineering, EDA,
machine learning, SQL analytics, and BI reporting for a simulated multi-category
e-commerce business.

---

## 1. Project Overview

This project builds a complete analytics solution for an e-commerce company,
taking a realistic 37,893-row transactional dataset through the full analyst
workflow: cleaning → exploration → feature engineering → customer
segmentation → predictive modeling → SQL analytics → BI-ready reporting.

Every number quoted in this repo (in the Excel workbook, the SQL query
results, and the Business Insights report) is computed directly from the
dataset checked into `/Dataset` — nothing is hardcoded or illustrative.

## 2. Business Problem

An e-commerce company's stakeholders need answers to five recurring
questions that currently require manual, ad-hoc analysis:

1. Which categories, products, and regions actually drive revenue and profit?
2. Which customers are most valuable, and which are at risk of churning?
3. Is the current discounting strategy helping or hurting margin?
4. Where are the operational bottlenecks (returns, shipping delays, cancellations)?
5. Can we predict churn and order value well enough to act on them proactively?

This project answers all five with reusable, re-runnable analytics assets
rather than one-off analysis.

## 3. Architecture

```
Raw CSV Tables (Dataset/)
        │
        ▼
Python Pipeline (Python/) ─── 01 Clean → 02 EDA → 03 Feature Engineer
        │                     → 04 Segment (RFM + KMeans)
        │                     → 05 Churn Model → 06 Sales Model
        ▼
   Cleaned / Engineered CSVs + Trained Models (Python/output/)
        │
        ├──────────────► Excel Workbook (AI_Ecommerce_Analysis.xlsx)
        │                 Dashboard, pivots, lookups, cleaning log
        │
        ├──────────────► SQL Database (SQL/)
        │                 Schema → Load → Indexes/Views → 42 Business Queries
        │
        └──────────────► Power BI / Reporting Layer
                          Executive, Sales, Customer, Product, AI, Finance pages

        ▼
Business_Insights.pdf  (52 data-driven insights + recommendations)
```

## 4. Tech Stack

| Layer | Tools |
|---|---|
| Data generation & cleaning | Python, Pandas, NumPy |
| Exploratory analysis | Matplotlib, Seaborn |
| Machine learning | Scikit-learn (Logistic Regression, Random Forest); XGBoost supported optionally |
| Database | MySQL 8.0+ (schema, views, indexes, stored procedures) |
| Spreadsheet analytics | Microsoft Excel (pivot-style formulas, conditional formatting, data validation, charts) |
| BI / dashboarding | Power BI |
| Reporting | ReportLab (PDF generation) |

## 5. Dataset Description

8 relational tables, 116,651 total rows, built with internally consistent
business logic (profit = sales − cost − shipping; churn = no order in 180+
days; CLV scales with actual spend) rather than pure randomness.

| Table | Rows | Description |
|---|---|---|
| `categories.csv` | 35 | Category → sub-category hierarchy |
| `products.csv` | 600 | Product catalog: cost, price, stock, rating |
| `customers.csv` | 6,000 | Demographics, segment, CLV, recency, churn label |
| `orders.csv` | 18,000 | Order header: status, totals, payment method |
| `order_items.csv` | 37,893 | Line-item detail: sales, profit, discount, returns |
| `payments.csv` | 18,000 | Transaction records |
| `shipping.csv` | 15,282 | Carrier, shipping mode, delivery time |
| `reviews.csv` | 20,841 | Review scores and text |

Date range: **July 2019 – July 2026**. Geography: 15 US states.

## 6. Folder Structure

```
├── Dataset/                       Raw CSV tables (see section 5)
├── Excel/
│   └── AI_Ecommerce_Analysis.xlsx Dashboard, pivots, lookup tool, cleaning log
├── Python/
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_customer_segmentation.py
│   ├── 05_churn_prediction.py
│   ├── 06_sales_prediction.py
│   └── output/                    Generated on run: cleaned data, figures, models
├── SQL/
│   ├── 01_create_schema.sql
│   ├── 02_load_data.sql
│   ├── 03_indexes_and_views.sql
│   └── 04_business_queries.sql    42 business questions with explanations
├── PowerBI/                        Build guide / .pbix (see section 9)
├── Screenshots/                    Dashboard screenshots
├── Business_Insights.pdf           52 insights + recommendations
├── requirements.txt
└── README.md
```

## 7. Key Results

**Churn Prediction Model** (`05_churn_prediction.py`)
Trained on genuinely predictive, leakage-free features only (Recency and
everything derived from it were deliberately excluded, since churn is
*defined* by recency in this dataset):
- Accuracy: ~63–64%
- ROC-AUC: ~0.66
- Output: 1,650 customers flagged High Risk for retention outreach

**Sales Prediction Model** (`06_sales_prediction.py`)
Regression on pre-sale-known features (quantity, price, discount, category,
customer profile):
- Random Forest R²: ~1.00 (expected — Sales is a deterministic function of
  Quantity × Price × (1−Discount); the model's practical value is as a
  data-quality check flagging any order line that doesn't fit the formula)
- Linear Regression baseline R²: 0.82

**Headline business metrics** (see `Business_Insights.pdf` for all 52):
- Total revenue: $22.27M | Total profit: $6.43M | Margin: 28.9%
- 84.3% of customers are repeat buyers
- Profit margin falls from 34.4% (no discount) to 6.5% on orders discounted 20%+
- 18.5% of shipments take longer than 7 days to arrive

## 8. Installation & Usage

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ai-ecommerce-analytics-platform

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the Python pipeline in order (each step writes to Python/output/)
cd Python
python 01_data_cleaning.py
python 02_eda.py
python 03_feature_engineering.py
python 04_customer_segmentation.py
python 05_churn_prediction.py
python 06_sales_prediction.py

# 4. Set up the SQL database (MySQL 8.0+)
mysql -u root -p < ../SQL/01_create_schema.sql
mysql -u root -p < ../SQL/02_load_data.sql      # update file paths first
mysql -u root -p < ../SQL/03_indexes_and_views.sql
mysql -u root -p < ../SQL/04_business_queries.sql

# 5. Open Excel/AI_Ecommerce_Analysis.xlsx directly - no setup required
```

## 9. Power BI Dashboard

The Power BI layer ships as a build guide (DAX measures + page-by-page
layout for Executive, Sales, Customer, Product, AI/Predictions, and Finance
pages) rather than a pre-built `.pbix`, since that file format can only be
authored inside Power BI Desktop itself. Follow the guide in `/PowerBI` to
reproduce the dashboard from the CSVs in `Python/output/` and `Dataset/`.

## 10. Screenshots

Dashboard and workbook screenshots live in `/Screenshots`.

## 11. Business Insights

See **`Business_Insights.pdf`** for the full report: 52 insights across
sales, profitability, customers, geography, products, pricing, seasonality,
logistics, and ML model takeaways, plus a Key Recommendations section.

## 12. Future Improvements

- Swap the 180-day static churn rule for a rolling/dynamic definition and
  retrain periodically
- Add time-series forecasting (e.g. Prophet/ARIMA) for demand planning at
  the category level, complementing the current per-order regression model
- Expand geographic coverage beyond the current 15 states
- Add a live Power BI dataflow connected directly to the SQL database instead
  of static CSV imports
- A/B test the discount-tier recommendation from the Business Insights report

## 13. License

This project uses synthetically generated data for educational and
portfolio purposes. Released under the MIT License — see `LICENSE` for
details.

## 14. Author

Built as an end-to-end data analytics portfolio project demonstrating the
complete Excel → Python → SQL → Power BI workflow.
