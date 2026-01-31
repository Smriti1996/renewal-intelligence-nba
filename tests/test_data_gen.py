# tests/test_data_gen.py

from pathlib import Path
from src.common.io import read_df


def test_member_features_exist():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    members_path = processed_dir / "members.parquet"

    assert members_path.exists(), "members.parquet not found; run data_gen first?"

    df = read_df(members_path)
    assert len(df) > 0
    assert "membership_nbr" in df.columns
