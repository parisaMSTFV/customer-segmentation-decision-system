# External Input Contract

`customer-segmentation segment` accepts one UTF-8 CSV with one row per customer. Contract version `1.0` requires a unique `customer_id` and these ten non-sensitive behavioral features:

| Field | Meaning | Validation |
|---|---|---|
| `recency_days` | Days since the latest completed order | Nonnegative |
| `orders_12m` | Completed orders in the last 12 months | Nonnegative |
| `revenue_12m` | Customer revenue in the last 12 months | Nonnegative |
| `margin_rate` | Contribution margin divided by revenue | 0 to 1 |
| `sessions_90d` | Sessions in the last 90 days | Nonnegative |
| `conversion_rate_90d` | Orders divided by eligible sessions | 0 to 1 |
| `discount_order_share` | Share of orders using a discount | 0 to 1 |
| `category_breadth_12m` | Distinct purchased categories in 12 months | Nonnegative |
| `return_rate` | Returned orders divided by completed orders | 0 to 1 |
| `satisfaction_score` | Consistently defined experience score | 1 to 5 |

Missing values, duplicate identifiers, and columns named `synthetic_persona`, `segment`, or `label` are rejected. Input should contain at least 60 customers; larger and temporally representative samples are strongly preferred.

## Command

```bash
customer-segmentation segment \
  --input /path/to/customer_features.csv \
  --output-dir artifacts/customer-segmentation
```

## Stable output

`customer_segments.csv` always contains:

```text
customer_id
cluster_id
segment_name
segment_definition_id
assignment_status
```

The `segment_definition_id` fingerprints fitted scaling parameters, centroids, and the relative business-name mapping. `assignment_status` is `descriptive_only`: campaign impact remains `Not evaluated` until an activation test measures it.

The run also writes profiles, a guarded decision playbook, two figures, and `run_metadata.json` with the input filename, SHA-256 checksum, contract version, segment size, and intrinsic silhouette. The input file itself is never copied into the output directory.
