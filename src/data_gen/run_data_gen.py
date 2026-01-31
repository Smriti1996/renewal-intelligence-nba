# src/data_gen/run_data_gen.py
from __future__ import annotations

from pathlib import Path

from src.common.utils import load_yaml, set_global_seed
from src.common.logging import setup_logger
from src.data_gen.generate_members import generate_members
from src.data_gen.generate_nba_uplift import generate_nba_uplift


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cfg_path = project_root / "configs" / "data_gen.yaml"

    cfg = load_yaml(cfg_path)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    logger = setup_logger("data_gen", log_dir=project_root / "data" / "logs")
    logger.info("Loaded config from %s", cfg_path)
    logger.info("Random seed set to %d", seed)

    files_cfg = cfg["files"]

    # 1. Generate members
    members_path = project_root / files_cfg["members_parquet"]
    logger.info("Generating members → %s", members_path)
    members_df = generate_members(cfg, output_path=members_path)
    logger.info("Members generated: %d rows", len(members_df))

    # 2. Generate NBA uplift summary
    nba_path = project_root / files_cfg["nba_uplift_parquet"]
    logger.info("Generating NBA uplift summary → %s", nba_path)
    nba_df = generate_nba_uplift(cfg, output_path=nba_path)
    logger.info("NBA uplift rows: %d", len(nba_df))

    logger.info("Data generation completed.")


if __name__ == "__main__":
    main()
