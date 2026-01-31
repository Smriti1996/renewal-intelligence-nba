# src/features/build_member_features.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from src.common.schema import MembersSchema


def _compute_engagement_bucket(sales_decile: pd.Series) -> pd.Series:
    """
    Map sales_decile to engagement bucket:
    1–3  -> low
    4–7  -> medium
    8–10 -> high
    """
    conditions = [
        sales_decile <= 3,
        (sales_decile >= 4) & (sales_decile <= 7),
        sales_decile >= 8,
    ]
    choices = ["low", "medium", "high"]
    return np.select(conditions, choices, default="medium")


def _compute_churn_risk_flag(
    tenure_months: pd.Series,
    auto_renew_opt_in: pd.Series,
    engagement_bucket: pd.Series,
) -> pd.Series:
    """
    Very simple risk heuristic:
    - High risk if:
        * tenure < 12 months AND
        * auto_renew_opt_in == 0 AND
        * engagement_bucket == 'low'
      else low risk.
    """
    high_risk = (
        (tenure_months < 12)
        & (auto_renew_opt_in == 0)
        & (engagement_bucket == "low")
    )
    return high_risk.astype(int)


def build_member_features(
    members_df: pd.DataFrame,
    cfg: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build a cleaned member_features table from raw members.

    Keeps the base membership/persona fields and adds:
    - engagement_bucket
    - churn_risk_flag
    """
    schema = MembersSchema()

    required_cols = schema.columns()
    missing = [c for c in required_cols if c not in members_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in members_df: {missing}")

    df = members_df.copy()

    # Engagement bucket from sales_decile
    df["engagement_bucket"] = _compute_engagement_bucket(df[schema.sales_decile])

    # Churn risk flag from tenure, auto-renew, engagement
    df["churn_risk_flag"] = _compute_churn_risk_flag(
        tenure_months=df[schema.tenure_months],
        auto_renew_opt_in=df[schema.auto_renew_opt_in],
        engagement_bucket=df["engagement_bucket"],
    )

    # You can add more derived features later; keep it simple for now.
    return df