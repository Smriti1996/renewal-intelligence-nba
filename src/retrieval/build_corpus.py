from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from src.common.schema import NbaUpliftSchema

def build_nba_fact_corpus(
        uplift_df: pd.DataFrame,
        cfg: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    
    s = NbaUpliftSchema()
    df = uplift_df.copy()

    if cfg is not None:
        min_support = cfg.get("min_support_for_corpus", 0)
        if min_support and min_support > 0:
            support = df[s.n_test_matched] + df[s.n_control_matched]
            df = df[support >= min_support]

    def format_fact(row) -> str:
        persona = int(row[s.persona_id])
        tenure = row[s.tenure_bucket]
        e_type = row[s.entity_type]
        e_name = row.get(s.entity_name, f"{e_type}_{int(row[s.entity_id])}")
        uplift = row[s.incremental_renewal_rate]
        uplift_bps = uplift * 10000.0
        test_rate = row[s.test_renewal_rate]
        control_rate = row[s.control_renewal_rate]
        n_test = int(row[s.n_test_matched])
        n_control = int(row[s.n_control_matched])

        return (
            f"For Persona {persona} members in tenure bucket '{tenure}', {e_type} "
            f"'{e_name}' shows an estimated incremental renewal uplift of about "
            f"{uplift_bps:.0f} basis points. In the synthetic-control analysis, the "
            f"renewal rate for exposed members was {test_rate:.1%}, compared with "
            f"{control_rate:.1%} for matched controls, based on {n_test} test and "
            f"{n_control} control members."
        )

    df["doc_id"] = (
        "nba_fact:"
        + df[s.persona_id].astype(str)
        + ":"
        + df[s.tenure_bucket].astype(str)
        + ":"
        + df[s.entity_type].astype(str)
        + ":"
        + df[s.entity_id].astype(str)
    )

    df["doc_type"] = "nba_fact"
    df["text"] = df.apply(format_fact, axis=1)

    corpus = df[
        [
            "doc_id",
            "doc_type",
            "text",
            s.persona_id,
            s.tenure_bucket,
            s.entity_type,
            s.entity_id,
            s.entity_name,
            s.incremental_renewal_rate,
            s.test_renewal_rate,
            s.control_renewal_rate,
            s.n_test_matched,
            s.n_control_matched,
        ]
    ].rename(
        columns={
            s.persona_id: "persona_id",
            s.tenure_bucket: "tenure_bucket",
            s.entity_type: "entity_type",
            s.entity_id: "entity_id",
            s.entity_name: "entity_name",
            s.incremental_renewal_rate: "incremental_renewal_rate",
            s.test_renewal_rate: "test_renewal_rate",
            s.control_renewal_rate: "control_renewal_rate",
            s.n_test_matched: "n_test_matched",
            s.n_control_matched: "n_control_matched",
        }
    )

    return corpus