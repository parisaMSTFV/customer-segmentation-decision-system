# Interview Guide

## One-minute explanation

I rebuilt a customer segmentation concept as a reproducible decision benchmark and governed scoring workflow. A deterministic generator creates observations and separate evaluator truth. Candidate evaluation uses development-only bootstrap resamples, absolute separation and stability floors, size, and an explicit business tolerance. Business names are also frozen on development profiles before holdout evaluation. A frozen ten-feature K-means pipeline is compared with fixed- and self-selected RFM baselines. External snapshots are fitted once, scored without refitting, pseudonymized by default, and monitored for feature drift, segment drift, centroid shift, and compatible migration. Actions remain hypotheses, not claimed campaign impact.

## Decisions worth explaining

- Why planted labels are excluded from fitting and used only for synthetic evaluation.
- Why an RFM-only model is a useful, understandable baseline.
- Why clustering quality needs separation, stability, size, and domain review rather than one score.
- Why protected demographics are excluded from this public decision example.
- Why a segment label is a summary of current behavior, not a causal diagnosis or permanent identity.
- Why an action playbook requires experiments before any ROI claim.
- Why a stable segment definition requires separate fit and score operations.
- Why failed size, semantic, feature PSI, segment PSI, or centroid-shift gates place activation on hold.

## Trade-offs

Six clusters are retained as a governed taxonomy only when eligible and within a declared tolerance of the best development score. This is a business choice supported by the synthetic benchmark, not a universal statistical requirement. A production rollout should use historical snapshots, review drift and migration, and validate treatment heterogeneity.
