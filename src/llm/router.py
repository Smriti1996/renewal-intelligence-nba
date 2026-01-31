# src/llm/router.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import pyarrow.parquet as pq

from src.common.utils import load_yaml
from src.common.io import read_df
from src.common.schema import MembersSchema
from src.common.logging import setup_logger

from src.retrieval.vector_store import VectorStore, VectorStoreConfig
from src.llm.ollama_client import OllamaClient
from src.llm import prompts


def _detect_intent(user_query: str, intents: list[str]) -> str:
    q = user_query.lower()

    if "why" in q or "reason" in q or "explain" in q:
        if "why_explanation" in intents:
            return "why_explanation"

    if "segment" in q or "persona" in q or "cohort" in q:
        if "segment_analysis" in intents:
            return "segment_analysis"

    if "graph" in q or "relation" in q or "connect" in q:
        if "kg_explore" in intents:
            return "kg_explore"

    if "next best" in q or "next-best" in q or "nba" in q or "next action" in q:
        if "member_nba" in intents:
            return "member_nba"

    return "general_help" if "general_help" in intents else intents[0]


# def _load_member_recos_for_member(
#     project_root: Path,
#     membership_nbr: Optional[int],
# ) -> List[Dict[str, Any]]:
#     """
#     Load recommendation rows only for a single member, using parquet filters
#     to avoid reading the entire file into memory.

#     Returns a list of dicts – safe to feed into the LLM.
#     """
#     if membership_nbr is None:
#         return []

#     app_cfg = load_yaml(project_root / "configs" / "app.yaml")
#     outputs_dir = Path(app_cfg["paths"]["outputs_dir"])
#     reco_path = project_root / outputs_dir / "member_nba_recos.parquet"

#     if not reco_path.exists():
#         return []

#     m_schema = MembersSchema()
#     membership_col = m_schema.membership_nbr

#     try:
#         # Use pyarrow directly to filter on the parquet side.
#         # This reads only the row groups that match, instead of the full file.
#         table = pq.read_table(
#             reco_path,
#             filters=[(membership_col, "==", membership_nbr)],
#         )
#         if table.num_rows == 0:
#             return []

#         # Narrow down to only the columns we actually care about for the LLM.
#         # Adjust these names to match your actual schema.
#         df = table.to_pandas()

#         # OPTIONAL: if the file has many columns, you can keep just a few:
#         # cols_to_keep = [membership_col, "persona_id", "entity_type", "entity_id",
#         #                 "incremental_renewal", "score", "rank"]
#         # df = df[[c for c in cols_to_keep if c in df.columns]]

#         # Convert to records; keep only a small number for safety
#         records = df.to_dict(orient="records")

#         # If there are many rows for this member, cap it for the LLM prompt
#         MAX_RECS = 50
#         return records[:MAX_RECS]

#     except Exception as e:
#         # Log the error but don't crash the process.
#         logger = setup_logger("llm_router", log_dir=project_root / "data" / "logs")
#         logger.error("Failed to load member recos for %s: %s", membership_nbr, e)
#         return []

def _load_member_recos_for_member(
    project_root: Path,
    membership_nbr: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Load recommendation rows only for a single member.

    Strategy:
    - Try a parquet-level filter via pyarrow (efficient).
    - If that returns 0 rows, fall back to pandas filtering.
    - Always cap rows to a small number for LLM context.
    """
    if membership_nbr is None:
        return []

    logger = setup_logger("llm_router", log_dir=project_root / "data" / "logs")
    app_cfg = load_yaml(project_root / "configs" / "app.yaml")
    outputs_dir = Path(app_cfg["paths"]["outputs_dir"])
    reco_path = project_root / outputs_dir / "member_nba_recos.parquet"

    if not reco_path.exists():
        logger.warning("Reco file not found at %s", reco_path)
        return []

    m_schema = MembersSchema()
    membership_col = m_schema.membership_nbr  # should be "membership_nbr"

    try:
        # First try: pyarrow filter (efficient)
        table = pq.read_table(
            reco_path,
            filters=[(membership_col, "==", membership_nbr)],
        )

        if table.num_rows > 0:
            df = table.to_pandas()
            logger.info(
                "Loaded %d member recos via pyarrow filter for %s",
                len(df),
                membership_nbr,
            )
        else:
            # Fallback: load full parquet and filter with pandas
            logger.info(
                "No rows via pyarrow filter for %s; falling back to pandas filter",
                membership_nbr,
            )
            df_full = pd.read_parquet(reco_path)
            # Handle possible type mismatch (str vs int)
            df = df_full[
                df_full[membership_col].astype(str) == str(membership_nbr)
            ].copy()
            logger.info(
                "Loaded %d member recos via pandas filter for %s",
                len(df),
                membership_nbr,
            )

        if df.empty:
            return []

        # Optional: limit to columns we actually care about in the prompt
        cols_to_keep = [
            membership_col,
            "engagement_bucket",
            "churn_risk_flag",
            "entity_type",
            "entity_id",
            "entity_name",
            "incremental_renewal_rate",
            "score",
            "member_rank",
            "explanation_short",
        ]
        df = df[[c for c in cols_to_keep if c in df.columns]]

        # Cap rows for LLM
        MAX_RECS = 50
        records = df.to_dict(orient="records")[:MAX_RECS]

        logger.info(
            "Returning %d member recos to LLM for %s",
            len(records),
            membership_nbr,
        )
        return records

    except Exception as e:
        logger.error("Failed to load member recos for %s: %s", membership_nbr, e)
        return []



def _load_vector_store(project_root: Path) -> VectorStore:
    app_cfg = load_yaml(project_root / "configs" / "app.yaml")
    llm_cfg = load_yaml(project_root / "configs" / "llm.yaml")

    retrieval_dir = Path(app_cfg["paths"]["retrieval_dir"])
    emb_cfg = llm_cfg.get("embedding", {})

    vs_cfg = VectorStoreConfig(
        model_name=emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        index_file=emb_cfg.get("index_file", "faiss_index.bin"),
        meta_file=emb_cfg.get("meta_file", "metadata.parquet"),
        normalize=bool(emb_cfg.get("normalize", True)),
    )
    store = VectorStore(cfg=vs_cfg, base_dir=retrieval_dir)
    store.load()
    return store


def _search_facts(
    store: VectorStore,
    query: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    return store.search(query=query, top_k=top_k)


def _fallback_no_llm_answer(
    user_query: str,
    intent: str,
    member_recos: List[Dict[str, Any]],
    retrieved_facts: List[Dict[str, Any]],
) -> str:
    """Simple text answer when Ollama is disabled."""
    lines = [
        "LLM is disabled in this environment.",
        f"Intent: {intent}",
        f"User query: {user_query}",
        f"Member recos loaded: {len(member_recos)}",
        f"Retrieved facts: {len(retrieved_facts)}",
    ]
    return "\n".join(lines)


def answer_query(
    user_query: str,
    membership_nbr: Optional[int] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    High-level router used by the API.
    This version is heavily instrumented for debugging.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    logger = setup_logger("llm_router", log_dir=project_root / "data" / "logs")

    logger.info("answer_query start, member=%s", membership_nbr)

    # Env flags
    debug_echo = os.getenv("RI_DEBUG_ECHO", "0") == "1"
    disable_retrieval = os.getenv("RI_DISABLE_RETRIEVAL", "0") == "1"
    disable_ollama = os.getenv("RI_DISABLE_OLLAMA", "0") == "1"
    disable_member_recos = os.getenv("RI_DISABLE_MEMBER_RECOS", "0") == "1"

    logger.info(
        "Flags: debug_echo=%s, disable_retrieval=%s, disable_ollama=%s, disable_member_recos=%s",
        debug_echo,
        disable_retrieval,
        disable_ollama,
        disable_member_recos,
    )

    if debug_echo:
        logger.info("Short-circuiting with DEBUG_ECHO")
        return {
            "answer": f"[DEBUG ECHO] You asked: {user_query} (member={membership_nbr})",
            "intent": "debug_echo",
            "used_member_nbr": membership_nbr,
        }

    # Load LLM config (with guard)
    try:
        llm_cfg = load_yaml(project_root / "configs" / "llm.yaml")
    except Exception as e:
        logger.exception("Failed to load llm.yaml: %s", e)
        return {
            "answer": f"Error loading LLM config: {e}",
            "intent": "config_error",
            "used_member_nbr": membership_nbr,
        }

    intents = llm_cfg.get("routing", {}).get("intents", [])
    if not intents:
        intents = ["general_help"]

    intent = _detect_intent(user_query, intents)
    logger.info("Detected intent: %s", intent)

    # ---------- Member recos (TEMPORARILY OPTIONAL) ----------
    member_recos: List[Dict[str, Any]] = []
    if membership_nbr is not None and intent in ("member_nba", "why_explanation"):
        if disable_member_recos:
            logger.info("Skipping member recos load (RI_DISABLE_MEMBER_RECOS=1)")
        else:
            logger.info("Loading member recos for membership_nbr=%s", membership_nbr)
            try:
                member_recos = _load_member_recos_for_member(project_root, membership_nbr)
                logger.info("Loaded %d member recos", len(member_recos))
            except Exception as e:
                logger.exception("Error loading member recos: %s", e)
                member_recos = []

    # ---------- Retrieval (facts) ----------
    retrieved_facts: List[Dict[str, Any]] = []
    if disable_retrieval:
        logger.info("Retrieval disabled via RI_DISABLE_RETRIEVAL")
    else:
        try:
            logger.info("Loading vector store and searching facts")
            store = _load_vector_store(project_root)
            retrieved_facts = _search_facts(store, user_query, top_k=10)
            logger.info("Retrieved %d facts", len(retrieved_facts))
        except Exception as e:
            logger.exception("Retrieval failed: %s", e)
            retrieved_facts = []

    # ---------- Build messages ----------
    logger.info("Building messages for intent=%s", intent)
    if intent == "member_nba":
        msgs = prompts.build_member_nba_messages(
            user_query=user_query,
            member_recos=member_recos,
            retrieved_facts=retrieved_facts,
        )
    elif intent == "segment_analysis":
        msgs = prompts.build_segment_analysis_messages(
            user_query=user_query,
            retrieved_facts=retrieved_facts,
        )
    elif intent == "why_explanation":
        msgs = prompts.build_why_explanation_messages(
            user_query=user_query,
            member_recos=member_recos,
            retrieved_facts=retrieved_facts,
        )
    elif intent == "kg_explore":
        msgs = prompts.build_kg_explore_messages(
            user_query=user_query,
            retrieved_facts=retrieved_facts,
        )
    else:
        msgs = prompts.build_general_help_messages(
            user_query=user_query,
            retrieved_facts=retrieved_facts,
        )

    logger.info("Messages built; count=%d", len(msgs))

    # ---------- LLM call or fallback ----------
    if disable_ollama:
        logger.info("RI_DISABLE_OLLAMA=1, using fallback answer")
        answer_text = _fallback_no_llm_answer(
            user_query=user_query,
            intent=intent,
            member_recos=member_recos,
            retrieved_facts=retrieved_facts,
        )
    else:
        logger.info("Calling Ollama client.chat()")
        try:
            client = OllamaClient.from_llm_config(project_root=project_root)
            answer_text = client.chat(messages=msgs)
        except Exception as e:
            logger.exception("Ollama call failed: %s", e)
            answer_text = _fallback_no_llm_answer(
                user_query=user_query,
                intent=intent,
                member_recos=member_recos,
                retrieved_facts=retrieved_facts,
            )

    logger.info("answer_query finished normally")
    return {
        "answer": answer_text,
        "intent": intent,
        "used_member_nbr": membership_nbr,
    }
