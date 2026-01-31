# src/reco/run_reco.py
from __future__ import annotations

from pathlib import Path

from src.common.utils import load_yaml
from src.common.io import read_df, write_df
from src.common.logging import setup_logger
from src.reco.candidate_gen import generate_member_candidates
from src.reco.scorer import score_candidates
from src.reco.ranker import rank_member_recos
from src.reco.explanations import add_explanations


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    app_cfg = load_yaml(project_root / "configs" / "app.yaml")
    data_gen_cfg = load_yaml(project_root / "configs" / "data_gen.yaml")
    reco_cfg = load_yaml(project_root / "configs" / "reco.yaml")

    logger = setup_logger("reco", log_dir=project_root / "data" / "logs")

    processed_dir = Path(app_cfg["paths"]["processed_dir"])
    raw_dir = Path(app_cfg["paths"]["raw_dir"])
    outputs_dir = Path(app_cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load inputs
    member_features_path = project_root / processed_dir / "member_features.parquet"
    nba_path = project_root / data_gen_cfg["files"]["nba_uplift_parquet"]

    if not member_features_path.exists():
        raise FileNotFoundError(
            f"member_features.parquet not found at {member_features_path}. "
            "Run `python -m src.features.run_features` first."
        )

    if not nba_path.exists():
        raise FileNotFoundError(
            f"nba_uplift_summary.parquet not found at {nba_path}. "
            "Run `python -m src.data_gen.run_data_gen` first."
        )

    logger.info("Loading member_features from %s", member_features_path)
    member_features_df = read_df(member_features_path)

    logger.info("Loading NBA uplift summary from %s", nba_path)
    uplift_df = read_df(nba_path)

    # 2) Generate candidates
    logger.info("Generating member-level candidates...")
    candidates_df = generate_member_candidates(
        member_features_df=member_features_df,
        uplift_df=uplift_df,
        reco_cfg=reco_cfg,
    )
    logger.info("Candidate rows: %d", len(candidates_df))

    # 3) Score candidates
    logger.info("Scoring candidates...")
    scored_df = score_candidates(candidates_df, reco_cfg=reco_cfg)

    # 4) Rank per member
    logger.info("Ranking top-K recommendations per member...")
    ranked_df = rank_member_recos(scored_df, reco_cfg=reco_cfg)

    # 5) Add explanations
    logger.info("Adding explanations...")
    explained_df = add_explanations(ranked_df)

    # 6) Write output
    output_path = project_root / outputs_dir / "member_nba_recos.parquet"
    logger.info("Writing member NBA recommendations to %s", output_path)
    write_df(explained_df, output_path, fmt="parquet")

    logger.info("Recommendation pipeline complete.")


if __name__ == "__main__":
    main()
