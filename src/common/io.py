from __future__ import annotations
from pathlib import Path
from typing import Literal

import pandas as pd


def ensure_parent_dir(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def write_df(df: pd.DataFrame, path: str | Path, fmt: Literal["parquet", "csv"] = "parquet") -> None:
    path = Path(path)
    ensure_parent_dir(path)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def read_df(path: str | Path, fmt: Literal["parquet", "csv"] | None = None) -> pd.DataFrame:
    path = Path(path)
    if fmt is None:
        if path.suffix == ".parquet":
            fmt = "parquet"
        elif path.suffix == ".csv":
            fmt = "csv"
        else:
            raise ValueError(f"Could not infer format from extension: {path}")

    if fmt == "parquet":
        return pd.read_parquet(path)
    elif fmt == "csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
