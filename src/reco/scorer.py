# src/reco/scorer.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from src.common.schema import MembersSchema, NbaUpliftSchema


def _normalize_series(x: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]. If constant, return zeros."""
    x_min = x.min()
    x_max = x.max()
    denom = x_max - x_min
    if denom == 0:
        return pd.Series(0.0, index=x.index)
    return (x - x_min) / denom


def _engagement_score(engagement_bucket: pd.Series) -> pd.Series:
    """
    engagement_bucket -> numeric:
    low -> 0.0
    medium -> 0.5
    high -> 1.0
    """
    mapping = {"low": 0.0, "medium": 0.5, "high": 1.0}
    return engagement_bucket.map(mapping).fillna(0.5)


def _risk_score(churn_risk_flag: pd.Series) -> pd.Series:
    """
    Simple risk score from churn_risk_flag:
    1 -> high risk (1.0)
    0 -> low risk (0.0)
    """
    return churn_risk_flag.astype(float)


def score_candidates(
    candidates_df: pd.DataFrame,
    reco_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Add a `score` column to member-level candidates using a simple
    weighted mixture of uplift, engagement, and risk.
    """
    m_schema = MembersSchema()
    u_schema = NbaUpliftSchema()

    df = candidates_df.copy()

    # Drop rows where entity info is missing (e.g., if no candidates for a segment)
    df = df[df[u_schema.entity_type].notna()]

    uplift_weight = float(reco_cfg.get("weights", {}).get("uplift_weight", 0.6))
    engagement_weight = float(
        reco_cfg.get("weights", {}).get("persona_affinity_weight", 0.3)
    )
    risk_weight = float(
        reco_cfg.get("weights", {}).get("recency_weight", 0.1)
    )  # reusing as risk weight

    # 1) Uplift in basis points, then normalized
    uplift_bps = df[u_schema.incremental_renewal_rate] * 10000.0
    uplift_norm = _normalize_series(uplift_bps)

    # 2) Engagement score from bucket
    if "engagement_bucket" not in df.columns:
        raise ValueError(
            "engagement_bucket not found in candidates_df. "
            "Make sure you used member_features as input."
        )
    engagement = _engagement_score(df["engagement_bucket"])

    # 3) Risk score from churn_risk_flag
    if "churn_risk_flag" not in df.columns:
        raise ValueError(
            "churn_risk_flag not found in candidates_df. "
            "Make sure you used member_features as input."
        )
    risk = _risk_score(df["churn_risk_flag"])

    # Final score = weighted sum
    df["score"] = (
        uplift_weight * uplift_norm
        + engagement_weight * engagement
        + risk_weight * risk
    )

    return df
