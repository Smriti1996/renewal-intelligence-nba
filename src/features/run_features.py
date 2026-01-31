# src/features/run_features.py
from __future__ import annotations

from pathlib import Path

from src.common.utils import load_yaml
from src.common.io import read_df, write_df
from src.common.logging import setup_logger
from src.features.build_member_features import build_member_features


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    data_gen_cfg = load_yaml(project_root / "configs" / "data_gen.yaml")
    app_cfg = load_yaml(project_root / "configs" / "app.yaml")

    logger = setup_logger("features", log_dir=project_root / "data" / "logs")

    raw_dir = Path(app_cfg["paths"]["raw_dir"])
    processed_dir = Path(app_cfg["paths"]["processed_dir"])

    members_path = project_root / data_gen_cfg["files"]["members_parquet"]
    if not members_path.exists():
        raise FileNotFoundError(
            f"members parquet not found at {members_path}. "
            "Run `python -m src.data_gen.run_data_gen` first."
        )

    logger.info("Loading members from %s", members_path)
    members_df = read_df(members_path)

    logger.info("Building member_features table...")
    member_features_df = build_member_features(members_df, cfg=data_gen_cfg)
    logger.info("member_features rows: %d", len(member_features_df))

    output_path = project_root / processed_dir / "member_features.parquet"
    logger.info("Writing member_features to %s", output_path)
    write_df(member_features_df, output_path, fmt="parquet")

    logger.info("Feature build complete.")


if __name__ == "__main__":
    main()
