-- ============================================================================
-- OlistTrust — analytical SQL queries
-- ----------------------------------------------------------------------------
-- These are the "business questions" answered purely in SQL against the SQLite
-- database built by src/etl/build_database.py. Each named query is loaded and
-- executed by src/etl/queries.py and surfaced in the dashboard.
-- ============================================================================


-- name: monthly_orders_revenue
-- Monthly order volume + revenue (delivered orders only).
SELECT
    strftime('%Y-%m', o.order_purchase_timestamp) AS month,
    COUNT(DISTINCT o.order_id)                     AS orders,
    ROUND(SUM(i.price), 2)                          AS revenue,
    ROUND(SUM(i.freight_value), 2)                  AS freight
FROM orders o
JOIN order_items i ON o.order_id = i.order_id
WHERE o.order_status = 'delivered'
GROUP BY month
ORDER BY month;


-- name: review_score_distribution
-- How customers rate their experience overall.
SELECT
    review_score,
    COUNT(*) AS n_reviews,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM order_reviews
GROUP BY review_score
ORDER BY review_score;


-- name: delivery_vs_satisfaction
-- The headline insight: late delivery destroys satisfaction.
-- Buckets orders by how they did vs the promised estimate.
SELECT
    CASE
        WHEN days_vs_estimate >= 5  THEN '1. Early (5+ days)'
        WHEN days_vs_estimate >= 0  THEN '2. On time'
        WHEN days_vs_estimate >= -5 THEN '3. Late (<5 days)'
        ELSE                              '4. Very late (5+ days)'
    END                          AS delivery_bucket,
    COUNT(*)                     AS n_orders,
    ROUND(AVG(review_score), 3)  AS avg_review_score,
    ROUND(100.0 * AVG(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END), 2) AS pct_negative
FROM v_order_core
WHERE review_score IS NOT NULL
  AND days_vs_estimate IS NOT NULL
GROUP BY delivery_bucket
ORDER BY delivery_bucket;


-- name: top_categories
-- Best-selling product categories (English names) with avg satisfaction.
SELECT
    COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category,
    COUNT(DISTINCT i.order_id)        AS orders,
    ROUND(SUM(i.price), 2)            AS revenue,
    ROUND(AVG(oc.review_score), 3)    AS avg_review_score
FROM order_items i
JOIN products p              ON i.product_id = p.product_id
LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
LEFT JOIN v_order_core oc    ON i.order_id = oc.order_id
GROUP BY category
HAVING orders >= 100
ORDER BY revenue DESC
LIMIT 20;


-- name: state_performance
-- Geographic view: orders, revenue and satisfaction by customer state.
SELECT
    c.customer_state                  AS state,
    COUNT(DISTINCT o.order_id)        AS orders,
    ROUND(AVG(oc.review_score), 3)    AS avg_review_score,
    ROUND(AVG(oc.delivery_days), 2)   AS avg_delivery_days
FROM orders o
JOIN customers c          ON o.customer_id = c.customer_id
LEFT JOIN v_order_core oc ON o.order_id = oc.order_id
GROUP BY state
ORDER BY orders DESC;


-- name: seller_leaderboard
-- Seller-level aggregates used as the backbone of the Trust Score.
-- Only sellers with a meaningful number of orders are kept.
SELECT
    so.seller_id,
    s.seller_state,
    COUNT(DISTINCT so.order_id)                                   AS n_orders,
    ROUND(AVG(so.review_score), 3)                               AS avg_review_score,
    ROUND(AVG(so.delivery_days), 2)                              AS avg_delivery_days,
    ROUND(AVG(so.days_vs_estimate), 2)                           AS avg_days_vs_estimate,
    ROUND(100.0 * AVG(CASE WHEN so.days_vs_estimate >= 0 THEN 1 ELSE 0 END), 2) AS on_time_rate,
    ROUND(100.0 * AVG(CASE WHEN so.review_score <= 2 THEN 1 ELSE 0 END), 2)     AS pct_negative,
    ROUND(SUM(so.price), 2)                                       AS revenue
FROM v_seller_orders so
LEFT JOIN sellers s ON so.seller_id = s.seller_id
WHERE so.review_score IS NOT NULL
GROUP BY so.seller_id
HAVING n_orders >= 10
ORDER BY avg_review_score DESC, n_orders DESC;
