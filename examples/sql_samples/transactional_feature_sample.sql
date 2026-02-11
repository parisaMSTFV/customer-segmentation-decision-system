/*
Transactional Feature Engineering Sample
----------------------------------------
Purpose: Build transaction/value signals per user for segmentation
Note: Schema and table names are generic and anonymized.
Assumptions:
- orders: one row per order
- order_items: one row per item in an order (with revenue & margin fields)
*/

WITH base_orders AS (
    SELECT
        o.user_id,
        o.order_id,
        o.order_date
    FROM orders o
    WHERE o.order_date >= DATEADD(month, -12, CURRENT_DATE)
      AND o.order_status = 'completed'
),

order_value AS (
    SELECT
        oi.order_id,
        SUM(oi.item_revenue) AS order_revenue,
        SUM(oi.item_margin)  AS order_margin
    FROM order_items oi
    GROUP BY oi.order_id
),

orders_enriched AS (
    SELECT
        bo.user_id,
        bo.order_id,
        bo.order_date,
        ov.order_revenue,
        ov.order_margin
    FROM base_orders bo
    LEFT JOIN order_value ov
        ON bo.order_id = ov.order_id
),

user_transactional AS (
    SELECT
        user_id,

        -- Volume / frequency
        COUNT(DISTINCT order_id) AS orders_12m,

        -- Monetary
        SUM(COALESCE(order_revenue, 0)) AS revenue_12m,
        SUM(COALESCE(order_margin, 0))  AS margin_12m,

        -- Basket metrics
        AVG(COALESCE(order_revenue, 0)) AS avg_order_value_12m,

        -- Recency
        DATEDIFF(day, MAX(order_date), CURRENT_DATE) AS recency_days,

        -- Purchase cycle (average days between orders)
        CASE
            WHEN COUNT(DISTINCT order_id) < 2 THEN NULL
            ELSE
                AVG(
                    DATEDIFF(
                        day,
                        LAG(order_date) OVER (PARTITION BY user_id ORDER BY order_date),
                        order_date
                    )
                )
        END AS avg_days_between_orders

    FROM orders_enriched
    GROUP BY user_id
)

SELECT *
FROM user_transactional;

