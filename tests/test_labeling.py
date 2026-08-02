from __future__ import annotations

import numpy as np
import pandas as pd

from customer_segmentation.labeling import (
    SEGMENT_ORDER,
    assign_business_names,
    build_decision_playbook,
    create_cluster_profiles,
    name_profiles,
)
from customer_segmentation.synthetic import generate_customers


def _profiles() -> pd.DataFrame:
    features, truth = generate_customers(900, 42)
    persona_codes = pd.Categorical(truth["synthetic_persona"]).codes
    return create_cluster_profiles(features.drop(columns="customer_id"), pd.Series(persona_codes))


def test_business_names_are_unique_and_complete() -> None:
    names = assign_business_names(_profiles())
    assert len(names) == 6
    assert set(names.values()) == set(SEGMENT_ORDER)


def test_playbook_has_action_and_guardrail_for_every_segment() -> None:
    profiles = _profiles()
    names = assign_business_names(profiles)
    playbook = build_decision_playbook(name_profiles(profiles, names))
    assert len(playbook) == 6
    assert playbook["testable_action"].str.len().gt(10).all()
    assert playbook["guardrail"].str.len().gt(10).all()
    assert playbook["impact_status"].str.startswith("Hypothesis").all()


def test_profile_counts_reconcile() -> None:
    profiles = _profiles()
    assert profiles["customers"].sum() == 900
    assert np.isclose(profiles["share"].sum(), 1.0)
