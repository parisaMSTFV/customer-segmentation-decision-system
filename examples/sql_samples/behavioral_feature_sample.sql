/*
Behavioral Feature Engineering Sample
--------------------------------------
Purpose: Extract behavioral engagement signals per user
Note: Schema and table names are generic and anonymized.
*/

WITH user_events AS (
    SELECT
        user_id,
        event_type,
        event_timestamp,
        session_id
    FROM events
    WHERE event_timestamp >= DATEADD(month, -3, CURRENT_DATE)
),

aggregated_behavior AS (
    SELECT
        user_id,

        -- Engagement intensity
        COUNT(CASE WHEN event_type = 'product_view' THEN 1 END) AS product_views_3m,
        COUNT(CASE WHEN event_type = 'add_to_cart' THEN 1 END) AS add_to_cart_3m,
        COUNT(DISTINCT session_id) AS sessions_3m,

        -- Engagement quality
        COUNT(CASE WHEN event_type = 'checkout_start' THEN 1 END) AS checkout_attempts_3m,

        -- Derived behavioral ratios
        CASE 
            WHEN COUNT(CASE WHEN event_type = 'product_view' THEN 1 END) = 0 THEN 0
            ELSE 
                COUNT(CASE WHEN event_type = 'add_to_cart' THEN 1 END) * 1.0 /
                COUNT(CASE WHEN event_type = 'product_view' THEN 1 END)
        END AS add_to_view_ratio

    FROM user_events
    GROUP BY user_id
)

SELECT *
FROM aggregated_behavior;
