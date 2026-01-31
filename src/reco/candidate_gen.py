# src/reco/candidate_gen.py
from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from src.common.schema import MembersSchema, NbaUpliftSchema


def _select_top_entities_per_segment(
    uplift_df: pd.DataFrame,
    max_candidates_per_segment: int,
) -> pd.DataFrame:
    """
    For each (persona_id, tenure_bucket), keep top-N entities by incremental_renewal_rate,
    across all entity types.
    """
    u_schema = NbaUpliftSchema()

    # Sort descending by uplift
    df = uplift_df.sort_values(
        by=[
            u_schema.persona_id,
            u_schema.tenure_bucket,
            u_schema.incremental_renewal_rate,
        ],
        ascending=[True, True, False],
    )

    # Rank within segment and keep top N
    df["segment_rank"] = (
        df.groupby([u_schema.persona_id, u_schema.tenure_bucket])[
            u_schema.incremental_renewal_rate
        ]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    df = df[df["segment_rank"] <= max_candidates_per_segment]

    return df


def generate_member_candidates(
    member_features_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    reco_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Generate member-level candidate NBAs:

    1. For each (persona_id, tenure_bucket), pick top entities by uplift.
    2. Join those entities onto all members in that segment.

    Returns a DataFrame with one row per (member, entity candidate).
    """
    m_schema = MembersSchema()
    u_schema = NbaUpliftSchema()

    max_candidates_per_segment = int(
        reco_cfg.get("max_candidates_per_segment", 20)
    )

    # 1) segment-level top entities
    segment_top = _select_top_entities_per_segment(
        uplift_df, max_candidates_per_segment=max_candidates_per_segment
    )

    # Keep only columns needed downstream
    segment_top = segment_top[
        [
            u_schema.persona_id,
            u_schema.tenure_bucket,
            u_schema.entity_type,
            u_schema.entity_id,
            u_schema.entity_name,
            u_schema.n_test_matched,
            u_schema.n_control_matched,
            u_schema.test_renewal_rate,
            u_schema.control_renewal_rate,
            u_schema.incremental_renewal_rate,
            "segment_rank",
        ]
    ]

    # 2) Join with members on (persona_id, tenure_bucket)
    # This assigns the segment's candidate entities to each member.
    merged = member_features_df.merge(
        segment_top,
        how="left",
        left_on=[m_schema.persona_id, m_schema.tenure_bucket],
        right_on=[u_schema.persona_id, u_schema.tenure_bucket],
        suffixes=("", "_uplift"),
    )

    # Drop duplicated join keys from uplift side
    merged = merged.drop(
        columns=[
            u_schema.persona_id,
            u_schema.tenure_bucket,
        ]
    )

    return merged
