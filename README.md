# Customer Segmentation Decision System

Decision-driven, multi-dimensional customer segmentation framework designed to guide targeting, marketing investment, and budget allocation in a large-scale e-commerce environment.

## Why this exists
Marketing investments become inefficient when customer value and behavior are not consistently defined.
This project designs a segmentation system that transforms fragmented customer signals into an interpretable decision layer for planning, personalization, and performance management.

## What this system does
- Builds **5 segmentation dimensions**: Demographics, Transactional, Behavioral, Psychographics, CX
- Produces dimension-level clusters and an integrated segmentation output
- Enables decision-making for:
  - targeting & personalization
  - budget allocation & ROI optimization
  - campaign effectiveness measurement
  - KPI monitoring and performance escalation

## Architecture (high-level)
1) Feature engineering (SQL)  
2) Dimension-level clustering (Python notebooks)  
3) Integration layer (multi-dimensional scoring / business rules)  
4) Visualization & stakeholder layer (BI)  
5) Monitoring & iteration (alerts + updates)

> Note: Production datasets, internal schemas, and operational dashboards are intentionally excluded.  
> This repository focuses on the **methodology, architecture, and safe examples**.

## Repository structure
- `docs/` Business context, solution overview, dimension design, modeling approach, impact validation
- `examples/sql_samples/` Safe SQL samples (schema-neutral)
- `examples/notebooks/` Demo clustering notebook (synthetic data)
- `artifacts/` Metric dictionary + segment definitions
- `presentation/` Case study summary (PDF)

## Contact
- LinkedIn: https://www.linkedin.com/in/parisa-mostafavi/
- Email: parisamostafavi23@gmail.com
