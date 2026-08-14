# Data Provenance

## Origin

All customer features and evaluator labels are generated locally from explicit persona parameters in `src/customer_segmentation/synthetic.py`. No production extract, customer identifier, internal table, or external dataset is used.

## Separation of observations and evaluator truth

`customer_features.csv` contains only the ten features available to the segmentation pipeline. The planted persona is written to `evaluator_truth.csv` and joined only after the development models have been frozen. A schema test rejects truth or label columns in model input.

## Evaluation split

The fixed seed creates 2,250 development customers and 750 holdout customers. Candidate cluster counts, scaling parameters, centroids, and business-name profiles are learned without using holdout truth. Synthetic truth ARI is then calculated on the holdout as an evaluator-only diagnostic.

## Appropriate use

This benchmark verifies code behavior, recovery of known synthetic structure, and reproducibility. Its accuracy does not estimate production performance, where feature definitions, missingness, seasonality, drift, and customer behavior would differ.

## External-input mode

The `segment` command accepts a local CSV through the versioned feature contract in [`INPUT_SCHEMA.md`](INPUT_SCHEMA.md). It records the input filename and SHA-256 checksum, but does not copy the source file into repository data or include evaluator truth. Its assignments are marked `descriptive_only`; action hypotheses still require controlled measurement.
