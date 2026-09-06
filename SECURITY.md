# Security and Privacy

## Public-data boundary

This repository contains synthetic customer records only. Do not commit real customer identifiers, events, transactions, feedback text, internal schema names, screenshots, credentials, endpoints, or local filesystem paths.

## Local inputs

If adapting the code to private data, keep inputs outside the repository and use access-controlled storage. External assignment IDs are HMAC-pseudonymized by default; keep `CUSTOMER_SEGMENTATION_ID_SALT` in an approved secret manager and rotate it deliberately. Pseudonyms remain personal data when they can be linked back to a person.

The input contract rejects unexpected columns so contact details, protected attributes, or evaluator labels are not silently carried into the workflow. Review aggregate profiles for small-cell and re-identification risk before sharing. Segment assignments must not be used to infer sensitive personal traits.

Fitted model artifacts use `joblib`, which can execute code while loading. Load artifacts only from a trusted, access-controlled source. The scoring workflow verifies the model checksum, definition checksum, fixed model filename, and exact runtime versions before deserialization. These checks detect corruption but do not authenticate a manifest controlled by an attacker.

## Automated check

`python scripts/check_sensitive.py` scans text artifacts for common credential formats, private network addresses, connection strings, and local user paths. It is a lightweight guardrail and does not replace a formal privacy review.

## Reporting issues

Open a private security report rather than a public issue if you believe sensitive information has been exposed.
