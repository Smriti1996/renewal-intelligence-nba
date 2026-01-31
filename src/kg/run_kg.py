# src/kg/run_kg.py
from __future__ import annotations

from pathlib import Path

from src.common.utils import load_yaml
from src.common.io import read_df
from src.common.logging import setup_logger
from src.kg.build_graph import build_kg_nodes, build_kg_edges


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    app_cfg = load_yaml(project_root / "configs" / "app.yaml")
    data_gen_cfg = load_yaml(project_root / "configs" / "data_gen.yaml")
    kg_cfg = load_yaml(project_root / "configs" / "kg.yaml")

    logger = setup_logger("kg", log_dir=project_root / "data" / "logs")

    raw_dir = Path(app_cfg["paths"]["raw_dir"])
    kg_dir = Path(app_cfg["paths"]["kg_dir"])
    kg_dir.mkdir(parents=True, exist_ok=True)

    nba_path = project_root / data_gen_cfg["files"]["nba_uplift_parquet"]
    if not nba_path.exists():
        raise FileNotFoundError(
            f"NBA uplift parquet not found at {nba_path}. "
            "Run `python -m src.data_gen.run_data_gen` first."
        )

    logger.info("Loading NBA uplift summary from %s", nba_path)
    uplift_df = read_df(nba_path)

    min_uplift = float(kg_cfg.get("kg", {}).get("min_uplift_for_edge", 0.0))
    logger.info("Building KG with min_uplift_for_edge=%.4f", min_uplift)

    logger.info("Building KG nodes...")
    nodes_df = build_kg_nodes(uplift_df)
    logger.info("KG nodes: %d", len(nodes_df))

    logger.info("Building KG edges...")
    edges_df = build_kg_edges(uplift_df, min_uplift_for_edge=min_uplift)
    logger.info("KG edges: %d", len(edges_df))

    nodes_path = project_root / kg_cfg["files"]["nodes_csv"]
    edges_path = project_root / kg_cfg["files"]["edges_csv"]

    logger.info("Writing KG nodes to %s", nodes_path)
    nodes_df.to_csv(nodes_path, index=False)

    logger.info("Writing KG edges to %s", edges_path)
    edges_df.to_csv(edges_path, index=False)

    logger.info("KG build complete.")


if __name__ == "__main__":
    main()
