# Security and Privacy

## Public-data boundary

This repository contains synthetic customer records only. Do not commit real customer identifiers, events, transactions, feedback text, internal schema names, screenshots, credentials, endpoints, or local filesystem paths.

## Local inputs

If adapting the code to private data, keep inputs outside the repository, use access-controlled storage, and review all generated profiles for re-identification risk before sharing. Segment assignments should not be used to infer sensitive personal traits.

## Automated check

`python scripts/check_sensitive.py` scans text artifacts for common credential formats, private network addresses, connection strings, and local user paths. It is a lightweight guardrail and does not replace a formal privacy review.

## Reporting issues

Open a private security report rather than a public issue if you believe sensitive information has been exposed.

