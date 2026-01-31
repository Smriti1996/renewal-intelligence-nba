# src/retrieval/run_retrieval.py
from __future__ import annotations

from pathlib import Path

from src.common.utils import load_yaml
from src.common.io import read_df
from src.common.logging import setup_logger
from src.retrieval.build_corpus import build_nba_fact_corpus
from src.retrieval.chunking import trivial_chunk_corpus, to_documents
from src.retrieval.vector_store import VectorStore, VectorStoreConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    app_cfg = load_yaml(project_root / "configs" / "app.yaml")
    data_gen_cfg = load_yaml(project_root / "configs" / "data_gen.yaml")
    retrieval_cfg = load_yaml(project_root / "configs" / "llm.yaml")

    logger = setup_logger("retrieval", log_dir=project_root / "data" / "logs")

    raw_dir = Path(app_cfg["paths"]["raw_dir"])
    retrieval_dir = Path(app_cfg["paths"]["retrieval_dir"])
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    nba_path = project_root / data_gen_cfg["files"]["nba_uplift_parquet"]
    if not nba_path.exists():
        raise FileNotFoundError(
            f"nba_uplift_summary.parquet not found at {nba_path}. "
            "Run `python -m src.data_gen.run_data_gen` first."
        )

    logger.info("Loading NBA uplift summary from %s", nba_path)
    uplift_df = read_df(nba_path)

    # 1) Build corpus
    logger.info("Building NBA fact corpus...")
    corpus_df = build_nba_fact_corpus(uplift_df, cfg=retrieval_cfg.get("corpus", {}))
    logger.info("Corpus size: %d docs", len(corpus_df))

    # 2) Chunk (no-op for now)
    corpus_df = trivial_chunk_corpus(corpus_df)

    # 3) Convert to docs
    docs = to_documents(corpus_df, text_col="text", id_col="doc_id")

    # 4) Build vector index
    model_name = retrieval_cfg.get("embedding", {}).get(
        "model_name", "sentence-transformers/all-MiniLM-L6-v2"
    )
    vs_cfg = VectorStoreConfig(
        model_name=model_name,
        index_file=retrieval_cfg.get("embedding", {}).get(
            "index_file", "faiss_index.bin"
        ),
        meta_file=retrieval_cfg.get("embedding", {}).get(
            "meta_file", "metadata.parquet"
        ),
        normalize=bool(retrieval_cfg.get("embedding", {}).get("normalize", True)),
    )

    logger.info("Building vector store with model '%s'", vs_cfg.model_name)
    store = VectorStore(cfg=vs_cfg, base_dir=retrieval_dir)
    store.build_from_docs(docs)
    store.save()

    logger.info("Retrieval index built and saved in %s", retrieval_dir)


if __name__ == "__main__":
    main()
