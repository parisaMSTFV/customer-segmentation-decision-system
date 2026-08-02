# Interview Guide

## One-minute explanation

I rebuilt a customer segmentation concept as a reproducible decision benchmark. A deterministic generator creates observations and separate evaluator truth. Model selection happens only on a development set using silhouette, seed stability, and a minimum-size guardrail. A frozen ten-feature K-means pipeline is compared with an RFM-only baseline on held-out synthetic customers. The result is translated into testable actions with measurement guardrails, not claimed campaign impact.

## Decisions worth explaining

- Why planted labels are excluded from fitting and used only for synthetic evaluation.
- Why an RFM-only model is a useful, understandable baseline.
- Why clustering quality needs separation, stability, size, and domain review rather than one score.
- Why protected demographics are excluded from this public decision example.
- Why a segment label is a summary of current behavior, not a causal diagnosis or permanent identity.
- Why an action playbook requires experiments before any ROI claim.

## Trade-offs

Six clusters narrowly lead the configured composite selection score. This is evidence for the synthetic benchmark, not a universal business requirement. A production rollout should repeat selection across time windows, assess migration and drift, review small segments, and validate treatment heterogeneity.

