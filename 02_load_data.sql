-- ================================================================================
-- 02_load_data.sql
-- --------------------------------------------------------------------------------
-- AI-Powered E-commerce Customer & Sales Analytics Platform
-- Loads the CSV files in /Dataset into the schema created by
-- 01_create_schema.sql, using MySQL's high-speed bulk loader.
--
-- Prerequisites:
--   1. Run 01_create_schema.sql first.
--   2. Enable local file loading (run once per session, or set in my.cnf):
--        SET GLOBAL local_infile = 1;
--   3. Update the file paths below to wherever /Dataset lives on your machine.
--   4. Load order matters (parent tables before child tables) because of the
--      foreign key constraints defined in the schema.
-- ================================================================================

USE ecommerce_analytics;

SET FOREIGN_KEY_CHECKS = 0;  -- temporarily relax FK order while bulk loading
SET GLOBAL local_infile = 1;

-- --------------------------------------------------------------------------------
-- 1. categories
-- --------------------------------------------------------------------------------
LOAD DATA LOCAL INFILE '/path/to/Dataset/categories.csv'
INTO TABLE categories
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Category_ID, Category, Sub_Category);

-- --------------------------------------------------------------------------------
-- 2. products
-- --------------------------------------------------------------------------------
LOAD DATA LOCAL INFILE '/path/to/Dataset/products.csv'
INTO TABLE products
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Product_ID, Product_Name, Category_ID, Category, Sub_Category, Brand,
 Unit_Cost, Unit_Price, Stock_Quantity, Reorder_Level, Product_Rating, Launch_Date);

-- --------------------------------------------------------------------------------
-- 3. customers
-- --------------------------------------------------------------------------------
LOAD DATA LOCAL INFILE '/path/to/Dataset/customers.csv'
INTO TABLE customers
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Customer_ID, First_Name, Last_Name, Gender, Customer_Age, Email, City, State,
 Country, Income_Group, Customer_Segment, Customer_Since, Total_Orders,
 Total_Spend, Customer_Lifetime_Value, Recency_Days, Churned);

-- --------------------------------------------------------------------------------
-- 4. orders
-- --------------------------------------------------------------------------------
LOAD DATA LOCAL INFILE '/path/to/Dataset/orders.csv'
INTO TABLE orders
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Order_ID, Customer_ID, Order_Date, @ship_date, Order_Status, @delivery_days,
 Total_Sales, Total_Profit, Item_Count, Payment_Method)
SET Ship_Date = NULLIF(@ship_date, ''),
    Delivery_Days = NULLIF(@delivery_days, '');

-- --------------------------------------------------------------------------------
-- 5. order_items
-- --------------------------------------------------------------------------------
LOAD DATA LOCAL INFILE '/path/to/Dataset/order_items.csv'
INTO TABLE order_items
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Order_Item_ID, Order_ID, Product_ID, Category, Sub_Category, Brand,
 Quantity, Unit_Price, Discount, Sales, Shipping_Cost, Profit, Return_Flag);

-- --------------------------------------------------------------------------------
-- 6. payments
-- --------------------------------------------------------------------------------
LOAD DATA LOCAL INFILE '/path/to/Dataset/payments.csv'
INTO TABLE payments
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Payment_ID, Order_ID, Payment_Method, Amount_Paid, Payment_Status, Transaction_Date);

-- --------------------------------------------------------------------------------
-- 7. shipping
-- --------------------------------------------------------------------------------
LOAD DATA LOCAL INFILE '/path/to/Dataset/shipping.csv'
INTO TABLE shipping
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Shipping_ID, Order_ID, Shipping_Mode, Carrier, Ship_Date, Delivery_Days, Shipping_City);

-- --------------------------------------------------------------------------------
-- 8. reviews
-- --------------------------------------------------------------------------------
LOAD DATA LOCAL INFILE '/path/to/Dataset/reviews.csv'
INTO TABLE reviews
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Review_ID, Order_Item_ID, Product_ID, Review_Score, Review_Text, @review_date)
SET Review_Date = NULLIF(@review_date, '');

SET FOREIGN_KEY_CHECKS = 1;

-- --------------------------------------------------------------------------------
-- Row-count sanity check after loading
-- --------------------------------------------------------------------------------
SELECT 'categories' AS table_name, COUNT(*) AS row_count FROM categories
UNION ALL SELECT 'products', COUNT(*) FROM products
UNION ALL SELECT 'customers', COUNT(*) FROM customers
UNION ALL SELECT 'orders', COUNT(*) FROM orders
UNION ALL SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL SELECT 'payments', COUNT(*) FROM payments
UNION ALL SELECT 'shipping', COUNT(*) FROM shipping
UNION ALL SELECT 'reviews', COUNT(*) FROM reviews;

-- Expected results (based on the shipped /Dataset CSVs):
--   categories    :    35
--   products      :   600
--   customers     : 6,000
--   orders        : 18,000
--   order_items   : 37,893
--   payments      : 18,000
--   shipping      : 15,282
--   reviews       : 20,841
