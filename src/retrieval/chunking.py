# src/retrieval/chunking.py
from __future__ import annotations

from typing import Iterable, List, Dict, Any
import pandas as pd


def trivial_chunk_corpus(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """
    No-op chunking: each row is already a short 'fact'.
    You can extend this later for longer docs.
    """
    return corpus_df.copy()


def to_documents(
    corpus_df: pd.DataFrame,
    text_col: str = "text",
    id_col: str = "doc_id",
) -> List[Dict[str, Any]]:
    """
    Turn a corpus DataFrame into a list of documents:
    {id, text, metadata}.
    """
    docs: List[Dict[str, Any]] = []
    for _, row in corpus_df.iterrows():
        meta = row.to_dict()
        text = meta.pop(text_col)
        doc_id = meta.pop(id_col)
        docs.append({"id": doc_id, "text": text, "metadata": meta})
    return docs
