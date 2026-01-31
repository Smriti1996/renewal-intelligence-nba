# tests/test_reco.py

from pathlib import Path

import pytest

from src.common.io import read_df


def test_member_nba_recos_schema():
    """
    Sanity check for the reco output file.

    - If member_nba_recos.parquet exists:
        * it should be non-empty
        * it should contain expected core columns

    - If it does not exist:
        * skip the test (run reco pipeline to enable this test)
    """
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "data" / "outputs"
    reco_path = outputs_dir / "member_nba_recos.parquet"

    if not reco_path.exists():
        pytest.skip(
            "member_nba_recos.parquet not found; run reco pipeline to enable this test."
        )

    df = read_df(reco_path)

    assert len(df) > 0, "member_nba_recos.parquet is empty."

    # Adjust to match your actual reco schema
    required_cols = [
        "membership_nbr",
        "entity_type",   # e.g. 'service', 'category', 'sub_category', 'action'
        "entity_id",
        "uplift_score",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"Reco output missing required columns: {missing}"
