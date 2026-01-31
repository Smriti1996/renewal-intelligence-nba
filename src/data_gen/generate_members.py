# src/data_gen/generate_members.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.common.constants import (
    TENURE_BUCKETS,
    MEMBERSHIP_TIERS,
    MEMBERSHIP_TYPES,
    SALES_DECILES,
    SALES_CENTILES,
)
from src.common.io import write_df
from src.common.schema import MembersSchema


def _sample_tenure_months(tenure_bucket: np.ndarray) -> np.ndarray:
    """Map tenure_bucket labels to approximate tenure_months."""
    # Simple ranges per bucket
    ranges = {
        "0-3m": (0, 3),
        "3-12m": (3, 12),
        "1-3y": (12, 36),
        "3y+": (36, 120),
    }
    mins = np.zeros_like(tenure_bucket, dtype=float)
    maxs = np.zeros_like(tenure_bucket, dtype=float)

    for label, (lo, hi) in ranges.items():
        mask = tenure_bucket == label
        mins[mask] = lo
        maxs[mask] = hi

    # Uniform within range, then round
    return np.round(np.random.uniform(mins, maxs)).astype(int)


def generate_members(cfg: Dict[str, Any], output_path: str | Path) -> pd.DataFrame:
    """
    Generate synthetic members data at membership_nbr level.

    The goal is to approximate:
    - persona distribution across members
    - tenure buckets + tenure_months
    - basic membership and sales attributes
    """
    n_members: int = int(cfg["n_members"])
    n_personas: int = int(cfg["n_personas"])
    tenure_buckets_cfg = cfg.get("tenure_buckets", TENURE_BUCKETS)

    schema = MembersSchema()

    # 1. Membership IDs (simple sequential IDs)
    membership_nbr = np.arange(1, n_members + 1, dtype=np.int64)

    # 2. Personas: assume roughly balanced with slight skew
    # e.g., more members in a few personas
    base_probs = np.ones(n_personas, dtype=float)
    base_probs[0] *= 1.2
    base_probs[1] *= 1.1
    base_probs = base_probs / base_probs.sum()

    persona_ids = np.random.choice(
        np.arange(1, n_personas + 1),
        size=n_members,
        p=base_probs,
    )

    # 3. Tenure buckets: rough distribution (more in 1-3y and 3y+)
    tb_labels = np.array(tenure_buckets_cfg)
    tb_probs = np.array([0.15, 0.30, 0.30, 0.25])
    tb_probs = tb_probs / tb_probs.sum()

    tenure_bucket = np.random.choice(tb_labels, size=n_members, p=tb_probs)

    # 4. Membership tier: maybe more Club than Plus
    tier_probs = np.array([0.7, 0.3])
    membership_tier = np.random.choice(
        MEMBERSHIP_TIERS, size=n_members, p=tier_probs
    )

    # 5. Membership type: Savings vs Business
    type_probs = np.array([0.8, 0.2])
    membership_type = np.random.choice(
        MEMBERSHIP_TYPES, size=n_members, p=type_probs
    )

    # 6. Auto renew opt-in: correlated with tenure (longer tenure → more opt-in)
    auto_renew_opt_in = np.zeros(n_members, dtype=int)
    for tb, prob in zip(tb_labels, [0.4, 0.55, 0.7, 0.8]):
        mask = tenure_bucket == tb
        auto_renew_opt_in[mask] = (
            np.random.rand(mask.sum()) < prob
        ).astype(int)

    # 7. Sales decile + centile: loosely correlated with tier and tenure
    # Higher tenure + Plus → higher spend, on average.
    # We'll just bias distributions a bit.
    sales_decile = np.random.choice(SALES_DECILES, size=n_members)
    sales_centile = np.random.choice(SALES_CENTILES, size=n_members)

    plus_mask = membership_tier == "Plus"
    high_tenure_mask = np.isin(tenure_bucket, ["1-3y", "3y+"])
    boost_mask = plus_mask & high_tenure_mask

    # Shift some boosted members to higher deciles/centiles
    sales_decile[boost_mask] = np.minimum(
        sales_decile[boost_mask] + np.random.randint(1, 3, size=boost_mask.sum()),
        max(SALES_DECILES),
    )
    sales_centile[boost_mask] = np.minimum(
        sales_centile[boost_mask] + np.random.randint(5, 20, size=boost_mask.sum()),
        max(SALES_CENTILES),
    )

    # 8. Tenure months derived from bucket
    tenure_months = _sample_tenure_months(tenure_bucket)

    df = pd.DataFrame(
        {
            schema.membership_nbr: membership_nbr,
            schema.persona_id: persona_ids,
            schema.tenure_bucket: tenure_bucket,
            schema.membership_tier: membership_tier,
            schema.membership_type: membership_type,
            schema.auto_renew_opt_in: auto_renew_opt_in,
            schema.sales_decile: sales_decile,
            schema.sales_centile: sales_centile,
            schema.tenure_months: tenure_months,
        }
    )

    write_df(df, output_path, fmt="parquet")
    return df
