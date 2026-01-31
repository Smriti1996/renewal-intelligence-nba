# tests/conftest.py

import sys
from pathlib import Path

# tests/ -> repo root
ROOT = Path(__file__).resolve().parents[1]

# Put repo root at the front of sys.path so `import src` works
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
