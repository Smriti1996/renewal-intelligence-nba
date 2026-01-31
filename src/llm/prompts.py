# src/llm/prompts.py
from __future__ import annotations

from typing import List, Dict, Any


SYSTEM_PROMPT = """You are a renewal intelligence assistant for a membership-based business.
You see:
- Member personas, tenure segments, engagement and churn risk
- Next-best-action (NBA) recommendations with incremental renewal uplift estimates
- A knowledge graph and fact corpus describing how categories, services and actions affect renewals

Rules:
- Be concise and clear. Explain in plain language, not jargon.
- Always ground answers in the provided facts and recommendations.
- If something is uncertain or not in the data, say so explicitly.
- Prefer practical, actionable suggestions over theory.
"""


def _format_facts_for_prompt(facts: list[dict[str, Any]]) -> str:
    if not facts:
        return "No retrieved facts."
    lines = []
    for i, f in enumerate(facts, start=1):
        txt = f.get("text") or ""
        persona = f.get("persona_id", "")
        tenure = f.get("tenure_bucket", "")
        entity_type = f.get("entity_type", "")
        entity_name = f.get("entity_name", "")
        lines.append(
            f"[Fact {i}] Persona={persona}, Tenure={tenure}, "
            f"Entity={entity_type}:{entity_name} -> {txt}"
        )
    return "\n".join(lines)


def _format_recos_for_prompt(recos: list[dict[str, Any]]) -> str:
    if not recos:
        return "No precomputed member recommendations."
    lines = []
    for r in recos:
        ent_type = r.get("entity_type")
        ent_name = r.get("entity_name")
        uplift = r.get("incremental_renewal_rate", 0.0) * 10000.0
        rank = r.get("member_rank")
        short_expl = r.get("explanation_short", "")
        lines.append(
            f"[Reco rank={rank}] {ent_type}:{ent_name} | uplift≈{uplift:.0f} bps | {short_expl}"
        )
    return "\n".join(lines)


def build_member_nba_messages(
    user_query: str,
    member_recos: list[dict[str, Any]],
    retrieved_facts: list[dict[str, Any]],
) -> list[dict[str, str]]:
    facts_block = _format_facts_for_prompt(retrieved_facts)
    recos_block = _format_recos_for_prompt(member_recos)

    user_content = (
        "User question:\n"
        f"{user_query}\n\n"
        "Precomputed top NBAs for this member:\n"
        f"{recos_block}\n\n"
        "Relevant renewal facts:\n"
        f"{facts_block}\n\n"
        "Using only the information above, answer the user's question, and clearly "
        "highlight 1–3 concrete next-best-actions with brief reasons."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_segment_analysis_messages(
    user_query: str,
    retrieved_facts: list[dict[str, Any]],
) -> list[dict[str, str]]:
    facts_block = _format_facts_for_prompt(retrieved_facts)

    user_content = (
        "User question:\n"
        f"{user_query}\n\n"
        "Segment-level renewal facts:\n"
        f"{facts_block}\n\n"
        "Summarize which actions, categories or services seem most important for the "
        "described segment, and explain the pattern in simple terms."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_why_explanation_messages(
    user_query: str,
    member_recos: list[dict[str, Any]],
    retrieved_facts: list[dict[str, Any]],
) -> list[dict[str, str]]:
    recos_block = _format_recos_for_prompt(member_recos)
    facts_block = _format_facts_for_prompt(retrieved_facts)

    user_content = (
        "The user is asking *why* certain actions or categories are being recommended.\n\n"
        f"User question:\n{user_query}\n\n"
        "Member recommendations:\n"
        f"{recos_block}\n\n"
        "Supporting renewal facts:\n"
        f"{facts_block}\n\n"
        "Explain in plain language why these are recommended, referring to uplift and "
        "relevant facts. If something is inferred, say 'based on the available data'."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_kg_explore_messages(
    user_query: str,
    retrieved_facts: list[dict[str, Any]],
) -> list[dict[str, str]]:
    facts_block = _format_facts_for_prompt(retrieved_facts)

    user_content = (
        "The user is asking about relationships between personas, tenure, and entities.\n\n"
        f"User question:\n{user_query}\n\n"
        "Available graph-related facts:\n"
        f"{facts_block}\n\n"
        "Describe the patterns and connections in simple terms. If the question asks for "
        "an action, suggest one based on the strongest positive facts."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_general_help_messages(
    user_query: str,
    retrieved_facts: list[dict[str, Any]],
) -> list[dict[str, str]]:
    facts_block = _format_facts_for_prompt(retrieved_facts)

    user_content = (
        "User question:\n"
        f"{user_query}\n\n"
        "Potentially relevant facts:\n"
        f"{facts_block}\n\n"
        "Answer the question as clearly as possible. If the question is outside "
        "renewal intelligence, say that your scope is renewal-related analytics and "
        "explain what you *can* help with."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
