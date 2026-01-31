# src/data_gen/generate_nba_uplift.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from src.common.constants import ENTITY_TYPES, TENURE_BUCKETS
from src.common.io import write_df
from src.common.schema import NbaUpliftSchema


def _build_entity_index(entity_specs: Dict[str, int]) -> pd.DataFrame:
    """
    Expand entity_specs into a full entity index:
    entity_type, entity_id, entity_name.
    """
    rows: List[dict] = []
    for e_type, count in entity_specs.items():
        for eid in range(1, count + 1):
            rows.append(
                {
                    "entity_type": e_type,
                    "entity_id": eid,
                    "entity_name": f"{e_type.upper()}_{eid:03d}",
                }
            )
    return pd.DataFrame(rows)


def generate_nba_uplift(cfg: Dict[str, Any], output_path: str | Path) -> pd.DataFrame:
    """
    Generate synthetic NBA uplift summary, mimicking a synthetic-control pipeline.

    Grain: persona_id, tenure_bucket, entity_type, entity_id
    """
    n_personas: int = int(cfg["n_personas"])
    tenure_buckets_cfg = cfg.get("tenure_buckets", TENURE_BUCKETS)
    entity_specs: Dict[str, int] = cfg["entity_specs"] if "entity_specs" in cfg else {}

    schema = NbaUpliftSchema()

    # 1. Build base entity index
    entity_idx = _build_entity_index(entity_specs)

    # Sanity: ensure we only use known entity types
    if not set(entity_idx["entity_type"]).issubset(set(ENTITY_TYPES)):
        unknown = set(entity_idx["entity_type"]) - set(ENTITY_TYPES)
        raise ValueError(f"Unknown entity types in config: {unknown}")

    rows: List[dict] = []

    # 2. For each persona × tenure × entity, simulate uplift metrics
    persona_ids = np.arange(1, n_personas + 1)
    tb_labels = np.array(tenure_buckets_cfg)

    for persona_id in persona_ids:
        for tb in tb_labels:
            # persona/tenure specific “base” renewal rate
            # e.g., early tenure slightly lower, long-tenure higher
            if tb == "0-3m":
                base_control = 0.55
            elif tb == "3-12m":
                base_control = 0.65
            elif tb == "1-3y":
                base_control = 0.8
            else:  # "3y+"
                base_control = 0.9

            # persona effect: some personas renew more easily
            persona_factor = 1.0 + (persona_id - 3) * 0.02  # small +/- shift
            base_control = np.clip(base_control * persona_factor, 0.4, 0.98)

            # For this (persona, tb), create metrics for all entities
            for _, erow in entity_idx.iterrows():
                e_type = erow["entity_type"]
                e_id = int(erow["entity_id"])
                e_name = erow["entity_name"]

                # Sample sizes: categories/sub-categories typically get more traffic
                if e_type == "action":
                    n_test = np.random.randint(500, 3000)
                    n_control = np.random.randint(500, 3000)
                elif e_type == "service":
                    n_test = np.random.randint(2000, 8000)
                    n_control = np.random.randint(2000, 8000)
                elif e_type == "category":
                    n_test = np.random.randint(5000, 20000)
                    n_control = np.random.randint(5000, 20000)
                else:  # sub_category
                    n_test = np.random.randint(2000, 15000)
                    n_control = np.random.randint(2000, 15000)

                # Control rate: jitter around base_control
                control_rate = np.clip(
                    np.random.normal(loc=base_control, scale=0.03), 0.3, 0.99
                )

                # Uplift: small positive or negative effect; slightly more positives
                uplift = np.random.normal(loc=0.01, scale=0.015)
                uplift = np.clip(uplift, -0.05, 0.08)

                test_rate = np.clip(control_rate + uplift, 0.3, 0.995)

                rows.append(
                    {
                        schema.persona_id: persona_id,
                        schema.tenure_bucket: tb,
                        schema.entity_type: e_type,
                        schema.entity_id: e_id,
                        schema.entity_name: e_name,
                        schema.n_test_matched: n_test,
                        schema.n_control_matched: n_control,
                        schema.test_renewal_rate: test_rate,
                        schema.control_renewal_rate: control_rate,
                        schema.incremental_renewal_rate: test_rate - control_rate,
                        # rank to be filled later
                        schema.incremental_rank: None,
                        schema.uplift_method: "synthetic_control",
                    }
                )

    df = pd.DataFrame(rows)

    # 3. Compute incremental_rank within each persona × tenure × entity_type
    df.sort_values(
        by=[
            schema.persona_id,
            schema.tenure_bucket,
            schema.entity_type,
            schema.incremental_renewal_rate,
        ],
        ascending=[True, True, True, False],
        inplace=True,
    )

    df[schema.incremental_rank] = (
        df.groupby(
            [schema.persona_id, schema.tenure_bucket, schema.entity_type]
        )[schema.incremental_renewal_rate]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    write_df(df, output_path, fmt="parquet")
    return df
