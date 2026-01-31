# src/reco/explanations.py
from __future__ import annotations

import pandas as pd

from src.common.schema import MembersSchema, NbaUpliftSchema


def add_explanations(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add short and long explanation strings for each recommendation row.

    These will later be used by the chatbot / LLM layer.
    """
    m_schema = MembersSchema()
    u_schema = NbaUpliftSchema()

    df = ranked_df.copy()

    # Safely access columns with defaults if missing
    persona_col = m_schema.persona_id if m_schema.persona_id in df.columns else None
    tenure_col = m_schema.tenure_bucket if m_schema.tenure_bucket in df.columns else None

    def short_expl(row) -> str:
        uplift_bps = row.get(u_schema.incremental_renewal_rate, 0.0) * 10000.0
        ent_name = row.get(u_schema.entity_name, "this action")
        p = row.get(persona_col, None)
        t = row.get(tenure_col, None)

        parts = []
        parts.append(f"{ent_name} shows a +{uplift_bps:.0f} bps uplift")

        if p is not None and t is not None:
            parts.append(f"for Persona {p}, {t} members")

        return "; ".join(parts)

    def long_expl(row) -> str:
        ent_name = row.get(u_schema.entity_name, "this action")
        uplift = row.get(u_schema.incremental_renewal_rate, 0.0)
        uplift_bps = uplift * 10000.0
        test_rate = row.get(u_schema.test_renewal_rate, None)
        control_rate = row.get(u_schema.control_renewal_rate, None)
        engagement = row.get("engagement_bucket", None)
        risk = row.get("churn_risk_flag", None)

        pieces = []

        pieces.append(
            f"This recommendation suggests focusing on '{ent_name}' because it is associated "
            f"with an estimated renewal uplift of about {uplift_bps:.0f} basis points for "
            f"similar members."
        )

        if (test_rate is not None) and (control_rate is not None):
            pieces.append(
                f"In the synthetic-control analysis, the renewal rate for members exposed to this "
                f"action was {test_rate:.1%} compared to {control_rate:.1%} for comparable controls."
            )

        if engagement is not None:
            pieces.append(
                f"The member is currently in the '{engagement}' engagement bucket, so this "
                f"action is positioned as an appropriate step to influence renewal."
            )

        if risk is not None:
            if int(risk) == 1:
                pieces.append(
                    "The member is flagged as higher churn risk, so interventions with strong "
                    "uplift are prioritized."
                )
            else:
                pieces.append(
                    "The member is not flagged as high churn risk, but this action still shows a "
                    "meaningful positive uplift."
                )

        return " ".join(pieces)

    df["explanation_short"] = df.apply(short_expl, axis=1)
    df["explanation_long"] = df.apply(long_expl, axis=1)

    return df
