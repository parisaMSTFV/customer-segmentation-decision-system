# External Input Contract

Contract version `2.0` accepts one UTF-8 CSV with one row per customer. Every row must use the same ISO `snapshot_date`; this makes the feature cutoff explicit and prevents a mixed-time snapshot from being scored as one population.

| Field | Meaning | Validation |
|---|---|---|
| `customer_id` | Stable customer identifier | Unique and nonblank |
| `snapshot_date` | Common feature cutoff | One `YYYY-MM-DD` value per file |
| `recency_days` | Days since the latest completed order | Nonnegative whole number |
| `orders_12m` | Completed orders in the last 12 months | Nonnegative whole number |
| `revenue_12m` | Customer revenue in the last 12 months | Nonnegative |
| `margin_rate` | Contribution margin divided by revenue | Signed value from -1 to 1 |
| `sessions_90d` | Eligible sessions in the last 90 days | Nonnegative whole number |
| `conversion_rate_90d` | Orders divided by eligible sessions | 0 to 1; zero when sessions are zero |
| `discount_order_share` | Smoothed share of orders using a discount | 0 to 1 |
| `category_breadth_12m` | Distinct purchased categories in 12 months | Nonnegative whole number |
| `return_rate` | Smoothed returned-order share | 0 to 1 |
| `satisfaction_score` | Consistently defined experience score | 1 to 5 |

Missing values, duplicate or blank identifiers, mixed snapshot dates, fractional counts, nonfinite values, and columns named `synthetic_persona`, `segment`, or `label` are rejected. Governed fitting requires at least 600 customers and fails when the six-segment taxonomy does not meet configured size, clipping, selection, and semantic-label guardrails.

## Fit a governed definition

```bash
customer-segmentation fit \
  --input /path/to/training_snapshot.csv \
  --model-dir artifacts/customer-segmentation/model \
  --output-dir artifacts/customer-segmentation/fit
```

The model directory contains an integrity-checked fitted pipeline and a JSON manifest. The definition identifier covers the ordered feature contract, preprocessing parameters, semantic centroids, training snapshot, configuration, and a row-order-invariant training-data fingerprint.

## Score a later snapshot

```bash
customer-segmentation score \
  --input /path/to/current_snapshot.csv \
  --model-dir artifacts/customer-segmentation/model \
  --output-dir artifacts/customer-segmentation/score \
  --previous-assignments /path/to/previous_customer_segments.csv
```

Scoring never refits the model. It verifies that the scoring date is not earlier than training, calculates segment-share PSI and standardized centroid shift, and optionally writes a customer migration matrix. Failed monitoring gates set every assignment to `review_required` and put the action playbook on hold.

## Stable assignment output

`customer_segments.csv` contains:

```text
customer_id
snapshot_date
cluster_id
segment_name
segment_definition_id
assignment_status
```

The input file is never copied into the output directory. Model artifacts use `joblib` and must only be loaded from a trusted local source.
