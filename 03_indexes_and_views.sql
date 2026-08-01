-- ================================================================================
-- 03_indexes_and_views.sql
-- --------------------------------------------------------------------------------
-- AI-Powered E-commerce Customer & Sales Analytics Platform
-- Performance indexes + reusable business views.
-- Run after 01_create_schema.sql and 02_load_data.sql.
-- ================================================================================

USE ecommerce_analytics;

-- --------------------------------------------------------------------------------
-- INDEXES
-- Primary keys are already indexed automatically. These target the columns
-- that the business queries in 04_business_queries.sql filter/join/group on
-- most often, to keep the analytics workload fast as the tables grow.
-- --------------------------------------------------------------------------------
CREATE INDEX idx_orders_customer      ON orders(Customer_ID);
CREATE INDEX idx_orders_date          ON orders(Order_Date);
CREATE INDEX idx_orders_status        ON orders(Order_Status);

CREATE INDEX idx_orderitems_order     ON order_items(Order_ID);
CREATE INDEX idx_orderitems_product   ON order_items(Product_ID);
CREATE INDEX idx_orderitems_category  ON order_items(Category);

CREATE INDEX idx_customers_state      ON customers(State);
CREATE INDEX idx_customers_segment    ON customers(Customer_Segment);
CREATE INDEX idx_customers_churned    ON customers(Churned);

CREATE INDEX idx_products_category    ON products(Category_ID);

CREATE INDEX idx_payments_order       ON payments(Order_ID);
CREATE INDEX idx_shipping_order       ON shipping(Order_ID);
CREATE INDEX idx_reviews_product      ON reviews(Product_ID);

-- --------------------------------------------------------------------------------
-- VIEW 1: vw_order_summary
-- One row per order-item, pre-joined with order/customer/product context.
-- This is the "flat table" most ad-hoc reporting queries will select from.
-- --------------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_order_summary AS
SELECT
    oi.Order_Item_ID,
    oi.Order_ID,
    o.Customer_ID,
    o.Order_Date,
    o.Order_Status,
    oi.Product_ID,
    oi.Category,
    oi.Sub_Category,
    oi.Brand,
    oi.Quantity,
    oi.Unit_Price,
    oi.Discount,
    oi.Sales,
    oi.Profit,
    oi.Return_Flag,
    c.State,
    c.City,
    c.Customer_Segment,
    c.Customer_Age,
    c.Gender
FROM order_items oi
JOIN orders o     ON oi.Order_ID = o.Order_ID
JOIN customers c  ON o.Customer_ID = c.Customer_ID;

-- --------------------------------------------------------------------------------
-- VIEW 2: vw_customer_ltv
-- Customer-level rollup: lifetime spend, order count, average order value,
-- and churn status - the single source of truth for customer-level KPIs.
-- --------------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_customer_ltv AS
SELECT
    c.Customer_ID,
    c.First_Name,
    c.Last_Name,
    c.State,
    c.Customer_Segment,
    c.Customer_Since,
    c.Churned,
    COUNT(DISTINCT o.Order_ID)           AS Total_Orders,
    ROUND(SUM(oi.Sales), 2)              AS Total_Sales,
    ROUND(SUM(oi.Profit), 2)             AS Total_Profit,
    ROUND(SUM(oi.Sales) / NULLIF(COUNT(DISTINCT o.Order_ID), 0), 2) AS Avg_Order_Value
FROM customers c
LEFT JOIN orders o      ON c.Customer_ID = o.Customer_ID
LEFT JOIN order_items oi ON o.Order_ID = oi.Order_ID
GROUP BY c.Customer_ID, c.First_Name, c.Last_Name, c.State,
         c.Customer_Segment, c.Customer_Since, c.Churned;

-- --------------------------------------------------------------------------------
-- VIEW 3: vw_monthly_sales
-- Month-level sales/profit trend - feeds the Monthly Trend chart in
-- Excel/Power BI directly without re-aggregating every time.
-- --------------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_monthly_sales AS
SELECT
    DATE_FORMAT(o.Order_Date, '%Y-%m')   AS Year_Month,
    COUNT(DISTINCT o.Order_ID)           AS Order_Count,
    ROUND(SUM(oi.Sales), 2)              AS Total_Sales,
    ROUND(SUM(oi.Profit), 2)             AS Total_Profit
FROM orders o
JOIN order_items oi ON o.Order_ID = oi.Order_ID
GROUP BY DATE_FORMAT(o.Order_Date, '%Y-%m');

-- --------------------------------------------------------------------------------
-- VIEW 4: vw_product_performance
-- Product-level sales, profit, return rate, and average review score.
-- --------------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_product_performance AS
SELECT
    p.Product_ID,
    p.Product_Name,
    p.Category,
    p.Sub_Category,
    p.Brand,
    p.Stock_Quantity,
    COALESCE(SUM(oi.Quantity), 0)                              AS Units_Sold,
    ROUND(COALESCE(SUM(oi.Sales), 0), 2)                       AS Total_Sales,
    ROUND(COALESCE(SUM(oi.Profit), 0), 2)                      AS Total_Profit,
    ROUND(
        SUM(CASE WHEN oi.Return_Flag = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / NULLIF(COUNT(oi.Order_Item_ID), 0), 1
    )                                                            AS Return_Rate_Pct,
    ROUND(AVG(r.Review_Score), 2)                               AS Avg_Review_Score
FROM products p
LEFT JOIN order_items oi ON p.Product_ID = oi.Product_ID
LEFT JOIN reviews r      ON p.Product_ID = r.Product_ID
GROUP BY p.Product_ID, p.Product_Name, p.Category, p.Sub_Category,
         p.Brand, p.Stock_Quantity;

-- --------------------------------------------------------------------------------
-- VIEW 5: vw_state_performance
-- State-level KPIs for the geographic pages of the Power BI dashboard.
-- --------------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_state_performance AS
SELECT
    c.State,
    COUNT(DISTINCT c.Customer_ID)   AS Customers,
    COUNT(DISTINCT o.Order_ID)      AS Orders,
    ROUND(SUM(oi.Sales), 2)         AS Total_Sales,
    ROUND(SUM(oi.Profit), 2)        AS Total_Profit
FROM customers c
JOIN orders o       ON c.Customer_ID = o.Customer_ID
JOIN order_items oi ON o.Order_ID = oi.Order_ID
GROUP BY c.State;

-- Quick verification that every view returns rows:
SELECT 'vw_order_summary' AS view_name, COUNT(*) AS row_count FROM vw_order_summary
UNION ALL SELECT 'vw_customer_ltv', COUNT(*) FROM vw_customer_ltv
UNION ALL SELECT 'vw_monthly_sales', COUNT(*) FROM vw_monthly_sales
UNION ALL SELECT 'vw_product_performance', COUNT(*) FROM vw_product_performance
UNION ALL SELECT 'vw_state_performance', COUNT(*) FROM vw_state_performance;
