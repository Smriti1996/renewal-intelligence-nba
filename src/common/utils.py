from __future__ import annotations
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
