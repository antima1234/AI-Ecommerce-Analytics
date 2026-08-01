-- ================================================================================
-- 01_create_schema.sql
-- --------------------------------------------------------------------------------
-- AI-Powered E-commerce Customer & Sales Analytics Platform
-- Database: MySQL 8.0+ compatible
--
-- Creates the full relational schema: 8 tables, primary keys, foreign keys,
-- and sensible column types/constraints derived from the actual dataset
-- (see /Dataset/*.csv for the data these tables are built to hold).
--
-- Load order matters because of foreign keys:
--   categories -> products -> customers -> orders -> order_items
--   -> payments -> shipping -> reviews
-- ================================================================================

DROP DATABASE IF EXISTS ecommerce_analytics;
CREATE DATABASE ecommerce_analytics
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE ecommerce_analytics;

-- --------------------------------------------------------------------------------
-- 1. CATEGORIES  (category / sub-category lookup)
-- --------------------------------------------------------------------------------
CREATE TABLE categories (
    Category_ID     VARCHAR(10)  NOT NULL,
    Category        VARCHAR(50)  NOT NULL,
    Sub_Category    VARCHAR(50)  NOT NULL,
    PRIMARY KEY (Category_ID)
) ENGINE=InnoDB;

-- --------------------------------------------------------------------------------
-- 2. PRODUCTS  (product catalog)
-- --------------------------------------------------------------------------------
CREATE TABLE products (
    Product_ID       VARCHAR(10)   NOT NULL,
    Product_Name     VARCHAR(150)  NOT NULL,
    Category_ID      VARCHAR(10)   NOT NULL,
    Category         VARCHAR(50)   NOT NULL,
    Sub_Category     VARCHAR(50)   NOT NULL,
    Brand            VARCHAR(50)   NOT NULL,
    Unit_Cost        DECIMAL(10,2) NOT NULL,
    Unit_Price       DECIMAL(10,2) NOT NULL,
    Stock_Quantity   INT           NOT NULL,
    Reorder_Level    INT           NOT NULL,
    Product_Rating   DECIMAL(2,1),
    Launch_Date      DATE,
    PRIMARY KEY (Product_ID),
    CONSTRAINT fk_products_category
        FOREIGN KEY (Category_ID) REFERENCES categories(Category_ID)
) ENGINE=InnoDB;

-- --------------------------------------------------------------------------------
-- 3. CUSTOMERS
-- --------------------------------------------------------------------------------
CREATE TABLE customers (
    Customer_ID               VARCHAR(10)   NOT NULL,
    First_Name                VARCHAR(50)   NOT NULL,
    Last_Name                 VARCHAR(50)   NOT NULL,
    Gender                    VARCHAR(20),
    Customer_Age              INT           NOT NULL,
    Email                     VARCHAR(150),
    City                      VARCHAR(50)   NOT NULL,
    State                     VARCHAR(50)   NOT NULL,
    Country                   VARCHAR(50)   NOT NULL,
    Income_Group              VARCHAR(20)   NOT NULL,
    Customer_Segment          VARCHAR(20)   NOT NULL,
    Customer_Since            DATE          NOT NULL,
    Total_Orders              INT           DEFAULT 0,
    Total_Spend                DECIMAL(12,2) DEFAULT 0,
    Customer_Lifetime_Value   DECIMAL(12,2) DEFAULT 0,
    Recency_Days              INT,
    Churned                   TINYINT(1)    DEFAULT 0,
    PRIMARY KEY (Customer_ID)
) ENGINE=InnoDB;

-- --------------------------------------------------------------------------------
-- 4. ORDERS  (order header)
-- --------------------------------------------------------------------------------
CREATE TABLE orders (
    Order_ID         VARCHAR(10)   NOT NULL,
    Customer_ID      VARCHAR(10)   NOT NULL,
    Order_Date       DATE          NOT NULL,
    Ship_Date        DATE,
    Order_Status     VARCHAR(20)   NOT NULL,
    Delivery_Days    INT,
    Total_Sales      DECIMAL(12,2) NOT NULL,
    Total_Profit     DECIMAL(12,2) NOT NULL,
    Item_Count       INT           NOT NULL,
    Payment_Method   VARCHAR(30)   NOT NULL,
    PRIMARY KEY (Order_ID),
    CONSTRAINT fk_orders_customer
        FOREIGN KEY (Customer_ID) REFERENCES customers(Customer_ID)
) ENGINE=InnoDB;

-- --------------------------------------------------------------------------------
-- 5. ORDER_ITEMS  (line-item fact table - grain of most analysis)
-- --------------------------------------------------------------------------------
CREATE TABLE order_items (
    Order_Item_ID    VARCHAR(12)   NOT NULL,
    Order_ID         VARCHAR(10)   NOT NULL,
    Product_ID       VARCHAR(10)   NOT NULL,
    Category         VARCHAR(50)   NOT NULL,
    Sub_Category     VARCHAR(50)   NOT NULL,
    Brand            VARCHAR(50)   NOT NULL,
    Quantity         INT           NOT NULL,
    Unit_Price       DECIMAL(10,2) NOT NULL,
    Discount         DECIMAL(4,2)  NOT NULL DEFAULT 0,
    Sales            DECIMAL(12,2) NOT NULL,
    Shipping_Cost    DECIMAL(8,2)  NOT NULL,
    Profit           DECIMAL(12,2) NOT NULL,
    Return_Flag      VARCHAR(3)    NOT NULL DEFAULT 'No',
    PRIMARY KEY (Order_Item_ID),
    CONSTRAINT fk_orderitems_order
        FOREIGN KEY (Order_ID) REFERENCES orders(Order_ID),
    CONSTRAINT fk_orderitems_product
        FOREIGN KEY (Product_ID) REFERENCES products(Product_ID)
) ENGINE=InnoDB;

-- --------------------------------------------------------------------------------
-- 6. PAYMENTS
-- --------------------------------------------------------------------------------
CREATE TABLE payments (
    Payment_ID         VARCHAR(10)   NOT NULL,
    Order_ID           VARCHAR(10)   NOT NULL,
    Payment_Method     VARCHAR(30)   NOT NULL,
    Amount_Paid        DECIMAL(12,2) NOT NULL,
    Payment_Status     VARCHAR(20)   NOT NULL,
    Transaction_Date   DATE          NOT NULL,
    PRIMARY KEY (Payment_ID),
    CONSTRAINT fk_payments_order
        FOREIGN KEY (Order_ID) REFERENCES orders(Order_ID)
) ENGINE=InnoDB;

-- --------------------------------------------------------------------------------
-- 7. SHIPPING
-- --------------------------------------------------------------------------------
CREATE TABLE shipping (
    Shipping_ID      VARCHAR(10)  NOT NULL,
    Order_ID         VARCHAR(10)  NOT NULL,
    Shipping_Mode    VARCHAR(20)  NOT NULL,
    Carrier          VARCHAR(30)  NOT NULL,
    Ship_Date        DATE         NOT NULL,
    Delivery_Days    INT          NOT NULL,
    Shipping_City    VARCHAR(50),
    PRIMARY KEY (Shipping_ID),
    CONSTRAINT fk_shipping_order
        FOREIGN KEY (Order_ID) REFERENCES orders(Order_ID)
) ENGINE=InnoDB;

-- --------------------------------------------------------------------------------
-- 8. REVIEWS
-- --------------------------------------------------------------------------------
CREATE TABLE reviews (
    Review_ID        VARCHAR(12)  NOT NULL,
    Order_Item_ID    VARCHAR(12)  NOT NULL,
    Product_ID       VARCHAR(10)  NOT NULL,
    Review_Score     INT          NOT NULL,
    Review_Text      VARCHAR(500),
    Review_Date      DATE,
    PRIMARY KEY (Review_ID),
    CONSTRAINT fk_reviews_orderitem
        FOREIGN KEY (Order_Item_ID) REFERENCES order_items(Order_Item_ID),
    CONSTRAINT fk_reviews_product
        FOREIGN KEY (Product_ID) REFERENCES products(Product_ID),
    CONSTRAINT chk_review_score CHECK (Review_Score BETWEEN 1 AND 5)
) ENGINE=InnoDB;

-- ================================================================================
-- Relationship summary
-- ================================================================================
-- categories (1) ----< products (many)
-- customers  (1) ----< orders (many)
-- orders     (1) ----< order_items (many)
-- products   (1) ----< order_items (many)
-- orders     (1) ----< payments (many, usually 1)
-- orders     (1) ----< shipping (many, usually 1)
-- order_items(1) ----< reviews (many, usually 0 or 1)
-- products   (1) ----< reviews (many)
-- ================================================================================
