# src/retrieval/vector_store.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class VectorStoreConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_file: str = "faiss_index.bin"
    meta_file: str = "metadata.parquet"
    normalize: bool = True


class VectorStore:
    def __init__(self, cfg: VectorStoreConfig, base_dir: Path) -> None:
        self.cfg = cfg
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.model: SentenceTransformer | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: pd.DataFrame | None = None

    # ---------------- Embeddings ---------------- #

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.cfg.model_name)
        return self.model

    def encode(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        emb = model.encode(texts, show_progress_bar=False)
        emb = np.asarray(emb, dtype="float32")
        if self.cfg.normalize:
            faiss.normalize_L2(emb)
        return emb

    # ---------------- Build / Persist ---------------- #

    def build_from_docs(self, docs: List[Dict[str, Any]]) -> None:
        texts = [d["text"] for d in docs]
        metas = [d["metadata"] | {"doc_id": d["id"]} for d in docs]

        embeddings = self.encode(texts)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.metadata = pd.DataFrame(metas)

    def save(self) -> None:
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index or metadata is None; build index first.")

        index_path = self.base_dir / self.cfg.index_file
        meta_path = self.base_dir / self.cfg.meta_file

        faiss.write_index(self.index, str(index_path))
        self.metadata.to_parquet(meta_path, index=False)

    def load(self) -> None:
        index_path = self.base_dir / self.cfg.index_file
        meta_path = self.base_dir / self.cfg.meta_file

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "Index or metadata not found. Run build step first."
            )

        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_parquet(meta_path)

    # ---------------- Query ---------------- #

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Search top_k documents for a query.
        Optional filters on metadata (exact match).
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index not loaded; call load() first.")

        # Optional filter subset
        meta = self.metadata
        if filters:
            for key, val in filters.items():
                if key in meta.columns:
                    meta = meta[meta[key] == val]

        if meta.empty:
            return []

        # Map filtered subset to index rows
        # We assume rows are in the same order as embeddings were added
        subset_indices = meta.index.to_numpy()
        # Encode query
        q_emb = self.encode([query])
        # Search full index then mask to subset for simplicity
        # (If performance is an issue later, build per-segment indices)
        scores, idxs = self.index.search(q_emb, top_k * 5)  # over-fetch then filter

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            if idx not in subset_indices:
                # skip if not in filtered metadata
                continue
            row = self.metadata.loc[idx].to_dict()
            row["score"] = float(score)
            results.append(row)
            if len(results) >= top_k:
                break

        return results
