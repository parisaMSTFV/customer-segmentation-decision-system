# Metric Dictionary

This document defines the core metrics used in the segmentation framework.  
Metrics are grouped by analytical dimension and documented to ensure clarity, consistency, and decision alignment.

---

## 1. Transactional Metrics

### orders_12m
Definition: Total number of completed orders in the past 12 months  
Purpose: Captures purchase frequency  
Business Use: Identifies loyal vs low-frequency customers  

---

### revenue_12m
Definition: Total gross revenue generated in the past 12 months  
Purpose: Measures monetary contribution  
Business Use: Investment prioritization and value ranking  

---

### margin_12m
Definition: Total margin contribution in the past 12 months  
Purpose: Profitability measurement  
Business Use: Budget allocation and ROI optimization  

---

### avg_order_value_12m
Definition: Average revenue per order in the past 12 months  
Purpose: Basket size indicator  
Business Use: Upsell and cross-sell targeting  

---

### recency_days
Definition: Days since last completed order  
Purpose: Activity freshness indicator  
Business Use: Churn detection and reactivation campaigns  

---

## 2. Behavioral Metrics

### sessions_3m
Definition: Total user sessions in the past 3 months  
Purpose: Engagement intensity  
Business Use: Detect active but low-converting users  

---

### product_views_3m
Definition: Total product page views in the past 3 months  
Purpose: Interest signal  
Business Use: Personalization and recommendation  

---

### add_to_cart_3m
Definition: Total add-to-cart events in the past 3 months  
Purpose: Purchase intent indicator  
Business Use: Conversion optimization  

---

### add_to_view_ratio
Definition: add_to_cart_3m divided by product_views_3m  
Purpose: Conversion efficiency proxy  
Business Use: Funnel quality evaluation  

---

## 3. Psychographic Metrics

### favorite_category_concentration
Definition: Percentage of interactions concentrated in top category  
Purpose: Preference intensity measurement  
Business Use: Persona identification  

---

### purchase_chain_pattern
Definition: Sequential purchase behavior across categories  
Purpose: Behavioral path identification  
Business Use: Cross-category targeting  

---

## 4. CX Metrics

### nps_score
Definition: Net Promoter Score reported by customer  
Purpose: Satisfaction indicator  
Business Use: Retention risk evaluation  

---

### return_rate
Definition: Percentage of returned or canceled orders  
Purpose: Experience friction measurement  
Business Use: Risk segmentation and quality control  

---

## Design Principles

- Metrics were engineered to be interpretable and decision-aligned  
- Avoided raw signals without business meaning  
- Balanced behavioral, financial, and engagement perspectives  
- Designed for scalability and periodic refresh  

Note: Production-level metric logic and data sources are excluded for confidentiality.
