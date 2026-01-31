# src/reco/ranker.py
from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from src.common.schema import MembersSchema, NbaUpliftSchema


def rank_member_recos(
    scored_df: pd.DataFrame,
    reco_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    For each member, sort candidates by score and keep top-K.

    Returns a DataFrame with one row per (member, recommended entity),
    ready for explanations.
    """
    m_schema = MembersSchema()
    u_schema = NbaUpliftSchema()

    top_k = int(reco_cfg.get("top_k_recos", 5))

    df = scored_df.copy()

    # Sort by (membership_nbr, score desc)
    df = df.sort_values(
        by=[m_schema.membership_nbr, "score"],
        ascending=[True, False],
    )

    # Rank within each member and keep top K
    df["member_rank"] = (
        df.groupby(m_schema.membership_nbr)["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    df = df[df["member_rank"] <= top_k]

    # Keep the most relevant columns
    keep_cols = [
        m_schema.membership_nbr,
        m_schema.persona_id,
        m_schema.tenure_bucket,
        "engagement_bucket",
        "churn_risk_flag",
        u_schema.entity_type,
        u_schema.entity_id,
        u_schema.entity_name,
        u_schema.incremental_renewal_rate,
        u_schema.test_renewal_rate,
        u_schema.control_renewal_rate,
        u_schema.n_test_matched,
        u_schema.n_control_matched,
        "segment_rank",
        "score",
        "member_rank",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]
