"""Input schema checks for synthetic customer features."""

from __future__ import annotations

import hashlib
import hmac
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT_COLUMN = "snapshot_date"
MAX_INPUT_BYTES = 250 * 1024 * 1024
MAX_IDENTIFIER_LENGTH = 128
FORMULA_PREFIXES = ("=", "+", "-", "@")
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")

FEATURE_COLUMNS = (
    "recency_days",
    "orders_12m",
    "revenue_12m",
    "margin_rate",
    "sessions_90d",
    "conversion_rate_90d",
    "discount_order_share",
    "category_breadth_12m",
    "return_rate",
    "satisfaction_score",
)
INPUT_COLUMNS = ("customer_id", SNAPSHOT_COLUMN, *FEATURE_COLUMNS)


class DataValidationError(ValueError):
    """Raised when customer features violate the public schema."""


def validate_customer_features(frame: pd.DataFrame) -> int:
    """Validate identifiers, temporal scope, feature types, bounds, and consistency."""
    required = set(INPUT_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"missing required columns: {missing}")
    unexpected = sorted(set(frame.columns).difference(required))
    if unexpected:
        raise DataValidationError(f"unexpected input columns are not allowed: {unexpected}")
    if frame.empty:
        raise DataValidationError("customer snapshot must not be empty")
    if frame["customer_id"].duplicated().any():
        raise DataValidationError("customer_id must be unique")
    if frame[list(required)].isna().any().any():
        raise DataValidationError("required columns must not contain null values")
    identifiers = frame["customer_id"].astype("string")
    if identifiers.str.strip().eq("").any():
        raise DataValidationError("customer_id must not be blank")
    if identifiers.ne(identifiers.str.strip()).any():
        raise DataValidationError("customer_id must not contain surrounding whitespace")
    if identifiers.str.len().gt(MAX_IDENTIFIER_LENGTH).any():
        raise DataValidationError(f"customer_id must not exceed {MAX_IDENTIFIER_LENGTH} characters")
    if identifiers.str.startswith(FORMULA_PREFIXES).any():
        raise DataValidationError("customer_id contains a spreadsheet-formula prefix")
    if identifiers.str.contains(CONTROL_CHARACTERS).any():
        raise DataValidationError("customer_id contains a control character")
    snapshot_values = frame[SNAPSHOT_COLUMN].astype("string")
    if snapshot_values.nunique(dropna=False) != 1:
        raise DataValidationError("all rows must use one snapshot_date")
    snapshot_date = str(snapshot_values.iloc[0])
    try:
        parsed_snapshot = date.fromisoformat(snapshot_date)
    except ValueError as error:
        raise DataValidationError("snapshot_date must use ISO YYYY-MM-DD format") from error
    if parsed_snapshot.isoformat() != snapshot_date:
        raise DataValidationError("snapshot_date must use ISO YYYY-MM-DD format")
    numeric_features = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric_features.isna().any().any() or not np.isfinite(numeric_features).all().all():
        raise DataValidationError("features must contain finite numeric values")
    integer_nonnegative = [
        "recency_days",
        "orders_12m",
        "sessions_90d",
        "category_breadth_12m",
    ]
    if (numeric_features[integer_nonnegative] < 0).any().any():
        raise DataValidationError("count features must be nonnegative")
    if not np.equal(numeric_features[integer_nonnegative] % 1, 0).all().all():
        raise DataValidationError("count features must contain whole numbers")
    if (numeric_features["revenue_12m"] < 0).any():
        raise DataValidationError("revenue_12m must be nonnegative")
    if not numeric_features["margin_rate"].between(-1, 1).all():
        raise DataValidationError("margin_rate must be between minus one and one")
    for column in ("conversion_rate_90d", "discount_order_share", "return_rate"):
        if not numeric_features[column].between(0, 1).all():
            raise DataValidationError(f"{column} must be between zero and one")
    if not numeric_features["satisfaction_score"].between(1, 5).all():
        raise DataValidationError("satisfaction_score must be between one and five")
    zero_sessions = numeric_features["sessions_90d"].eq(0)
    if numeric_features.loc[zero_sessions, "conversion_rate_90d"].ne(0).any():
        raise DataValidationError("customers with zero sessions must have zero conversion rate")
    return 15


def _read_string_csv(path: Path) -> pd.DataFrame:
    """Read a bounded UTF-8 CSV without coercing customer identifiers."""
    if not path.is_file():
        raise DataValidationError(f"CSV file does not exist: {path}")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise DataValidationError("CSV exceeds the 250 MB input limit")
    try:
        return pd.read_csv(path, dtype="string", keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DataValidationError("input must be a readable UTF-8 CSV with a header") from error


def load_customer_snapshot(path: Path) -> tuple[pd.DataFrame, int, str]:
    """Load and validate one exact customer-feature snapshot."""
    frame = _read_string_csv(path)
    checks_passed = validate_customer_features(frame)
    canonical = frame.copy()
    for column in FEATURE_COLUMNS:
        canonical[column] = pd.to_numeric(canonical[column], errors="raise")
    return canonical, checks_passed, str(canonical[SNAPSHOT_COLUMN].iloc[0])


def validate_identifier_policy(policy: str, salt: str | None) -> None:
    """Fail closed unless exports use a supported identifier policy."""
    if policy not in {"pseudonymized", "raw"}:
        raise DataValidationError("identifier_policy must be pseudonymized or raw")
    if policy == "pseudonymized" and (salt is None or len(salt) < 16):
        raise DataValidationError(
            "A salt of at least 16 characters is required for pseudonymized exports; "
            "set CUSTOMER_SEGMENTATION_ID_SALT"
        )


def pseudonymize_customer_ids(frame: pd.DataFrame, salt: str) -> pd.DataFrame:
    """Replace customer IDs with deterministic HMAC-SHA256 tokens."""
    validate_identifier_policy("pseudonymized", salt)

    def token(value: object) -> str:
        digest = hmac.new(
            salt.encode("utf-8"),
            f"customer:{value}".encode(),
            hashlib.sha256,
        ).hexdigest()[:20]
        return f"cus_{digest}"

    safe = frame.copy()
    safe["customer_id"] = safe["customer_id"].map(token)
    return safe


def load_previous_assignments(path: Path) -> pd.DataFrame:
    """Load an earlier assignment export used for migration monitoring."""
    frame = _read_string_csv(path)
    required = {
        "customer_id",
        SNAPSHOT_COLUMN,
        "segment_name",
        "segment_definition_id",
        "identifier_policy",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"previous assignments are missing columns: {missing}")
    if frame.empty:
        raise DataValidationError("previous assignments must not be empty")
    if frame["customer_id"].duplicated().any():
        raise DataValidationError("previous assignments contain duplicate customer_id values")
    if frame["customer_id"].str.strip().eq("").any():
        raise DataValidationError("previous assignment customer_id must not be blank")
    for column in (SNAPSHOT_COLUMN, "segment_definition_id", "identifier_policy"):
        if frame[column].nunique(dropna=False) != 1:
            raise DataValidationError(f"previous assignments must contain one {column}")
    if frame["segment_name"].str.strip().eq("").any():
        raise DataValidationError("previous assignment segment_name must not be blank")
    snapshot_date = str(frame[SNAPSHOT_COLUMN].iloc[0])
    try:
        parsed_snapshot = date.fromisoformat(snapshot_date)
    except ValueError as error:
        raise DataValidationError(
            "previous assignment snapshot_date must use ISO YYYY-MM-DD format"
        ) from error
    if parsed_snapshot.isoformat() != snapshot_date:
        raise DataValidationError(
            "previous assignment snapshot_date must use ISO YYYY-MM-DD format"
        )
    return frame
