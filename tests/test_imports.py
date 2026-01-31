# tests/test_imports.py

from importlib import import_module


MODULES = [
    "src.common.logging",
    "src.common.io",
    "src.common.utils",
    "src.common.schema",
    "src.llm.ollama_client",
    "src.llm.prompts",
    "src.llm.router",
    "src.retrieval.vector_store",
    "src.api.main",
    "src.api.routes",
    "src.ui.streamlit_app",
]


def test_imports_smoke():
    """
    Light-weight smoke test to ensure core modules import correctly.
    This catches syntax errors and missing dependencies early in CI.
    """
    for module_name in MODULES:
        import_module(module_name)
