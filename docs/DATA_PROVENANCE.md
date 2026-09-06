# Data Provenance

## Origin

All customer features and evaluator labels are generated locally from explicit persona parameters in `src/customer_segmentation/synthetic.py`. No production extract, customer identifier, internal table, or external dataset is used.

## Separation of observations and evaluator truth

`customer_features.csv` contains only the ten features available to the segmentation pipeline. The planted persona is written to `evaluator_truth.csv` and joined only after the development models have been frozen. The external schema rejects every unexpected column, including evaluator labels and sensitive attributes.

## Evaluation split

The fixed seed creates 2,250 development customers and 750 holdout customers. Candidate cluster counts, clipping and scaling parameters, centroids, and business-name profiles are learned without using holdout truth. Synthetic truth ARI is then calculated on the holdout as an evaluator-only diagnostic.

## Appropriate use

This benchmark verifies code behavior, recovery of known synthetic structure, and reproducibility. Its accuracy does not estimate production performance, where feature definitions, missingness, seasonality, drift, and customer behavior would differ.

## External-input mode

The `fit` command creates a versioned, checksum-verified model from one dated snapshot. The `score` command applies that frozen definition to the same or a later snapshot without refitting, calculates feature and segment drift, and can compare compatible assignments from a prior period. Source files remain local, exported IDs are pseudonymized by default, and action hypotheses still require controlled measurement.
