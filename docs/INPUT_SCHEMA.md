# External Input Contract

Contract version `3.0` accepts one UTF-8 CSV with exactly one row per customer. Every row must use
the same ISO `snapshot_date`; this makes the feature cutoff explicit and prevents a mixed-time
population from being scored as one snapshot.

| Field | Meaning | Validation |
|---|---|---|
| `customer_id` | Stable customer identifier | Unique, nonblank string; maximum 128 characters |
| `snapshot_date` | Common feature cutoff | One exact `YYYY-MM-DD` value per file |
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

Identifiers are parsed as strings before numeric conversion, preserving Unicode and leading zeroes.
Values with surrounding whitespace, control characters, spreadsheet-formula prefixes (`=`, `+`,
`-`, or `@`), or more than 128 characters are rejected. Missing values, duplicate IDs, mixed dates,
fractional counts, nonfinite values, and unexpected columns are also rejected. The input must be no
larger than 250 MB.

Governed fitting requires at least 600 customers. The six-segment taxonomy must satisfy minimum
size, silhouette, resample-stability, clipping, candidate-selection, and semantic-label guardrails.

## Default pseudonymized export

External assignments use deterministic HMAC-SHA256 customer tokens by default. Supply a secret of
at least 16 characters through `CUSTOMER_SEGMENTATION_ID_SALT`; the secret is never written to an
artifact.

```bash
export CUSTOMER_SEGMENTATION_ID_SALT="$(python -c 'import secrets; print(secrets.token_hex(32))')"
customer-segmentation fit \
  --input /path/to/training_snapshot.csv \
  --model-dir artifacts/customer-segmentation/model \
  --output-dir artifacts/customer-segmentation/fit
```

Keep the same secret in an approved secret manager when assignments must be linked across
snapshots. HMAC tokens are pseudonymous data, not anonymous data, and remain subject to the relevant
access and retention policy.

Raw identifiers can be exported only with `--allow-raw-identifiers`. This override is intended for
approved public or fully synthetic inputs.

## Frozen model definition

The model directory contains a `joblib` pipeline and JSON manifest. The manifest records the exact
package, NumPy, and scikit-learn versions; scoring fails when they differ. It also binds the model
checksum, semantic definition, ordered feature contract, preprocessing parameters, training date,
governance thresholds, and a row-order-invariant training-data fingerprint.
Scoring also fails if its active governance config differs from the policy frozen into that
definition.

The checksum detects corruption or accidental replacement; it does not authenticate an attacker-
controlled manifest. Because `joblib` can execute code while loading, use model artifacts only from
a trusted, access-controlled source.

## Score a current snapshot

```bash
customer-segmentation score \
  --input /path/to/current_snapshot.csv \
  --model-dir artifacts/customer-segmentation/model \
  --output-dir artifacts/customer-segmentation/score \
  --previous-assignments /path/to/previous/customer_segments.csv \
  --fail-on-review
```

Scoring never refits the model. It verifies that the scoring date is not earlier than training and
calculates:

- PSI for every model feature using frozen training quantile bins;
- PSI for governed segment shares;
- standardized centroid shift and clipping share;
- optional customer migration for overlapping IDs.

Previous assignments must use the same `segment_definition_id` and identifier policy, must contain
one valid snapshot date that is not later than the current snapshot, and must share customers with
the current export. A different HMAC secret produces no overlap and therefore fails closed.

Failed monitoring gates set every assignment to `review_required` and put the action playbook on
hold. `--fail-on-review` also returns process exit code 2 for scheduled workflows.

## Stable assignment output

`customer_segments.csv` contains:

```text
customer_id
snapshot_date
cluster_id
segment_name
segment_definition_id
assignment_status
identifier_policy
```

The source CSV is never copied and its filename is not written to output metadata. Aggregate
profiles contain no customer identifiers.
