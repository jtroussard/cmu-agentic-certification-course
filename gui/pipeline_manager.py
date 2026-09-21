"""
gui/pipeline_manager.py

Singleton wrapper that loads all three heavyweight objects (extraction model,
verification model, embedding function) exactly once per Streamlit session and
caches them in st.session_state.  Subsequent calls return the cached instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from pipeline import MailOrganizerPipeline

# ---------------------------------------------------------------------------
# Defaults (mirrors run.py)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_PATH = _ROOT / "models" / "qwen2.5-7b-instruct-q4_k_m.gguf"
DEFAULT_DB_PATH = _ROOT / "data" / "mail_organizer.db"

_PIPELINE_KEY = "_mop_pipeline"
_STATUS_KEY = "_mop_status"   # "unloaded" | "loading" | "ready" | "error"
_ERROR_KEY = "_mop_error"


# Development mode flag: when True, NEVER caches the pipeline and reloads all project modules on disk
DEV_MODE: bool = True


def get_pipeline(
    model_path: Path | None = None,
    db_path: Path | None = None,
    n_gpu_layers: int = -1,
) -> "MailOrganizerPipeline | None":
    """
    Return a freshly initialized pipeline with latest code reloaded from disk.
    In DEV_MODE, does not cache the pipeline in st.session_state.
    """
    model_path = model_path or DEFAULT_MODEL_PATH
    db_path = db_path or DEFAULT_DB_PATH

    st.session_state[_STATUS_KEY] = "loading"

    try:
        import sys
        import importlib

        # In DEV_MODE, dynamically reload all project modules so changes on disk take immediate effect
        if DEV_MODE:
            project_prefixes = ("agents.", "models.", "pipeline", "ingestion.", "rag.", "db.")
            for mod_name in list(sys.modules.keys()):
                if any(mod_name.startswith(p) for p in project_prefixes):
                    try:
                        importlib.reload(sys.modules[mod_name])
                    except Exception:
                        pass

        from models.llm_adapter import (
            build_embedding_fn,
            build_extraction_model,
            build_verification_model,
        )
        from pipeline import MailOrganizerPipeline

        extraction_model = build_extraction_model(model_path, n_gpu_layers=n_gpu_layers)
        verification_model = build_verification_model(model_path, n_gpu_layers=n_gpu_layers)
        embed_fn = build_embedding_fn()

        pipeline = MailOrganizerPipeline(
            extraction_model=extraction_model,
            verification_model=verification_model,
            embed_fn=embed_fn,
            db_path=db_path,
        )

        if not DEV_MODE:
            st.session_state[_PIPELINE_KEY] = pipeline
        else:
            st.session_state.pop(_PIPELINE_KEY, None)

        st.session_state[_STATUS_KEY] = "ready"
        return pipeline

    except Exception as exc:  # noqa: BLE001
        st.session_state[_STATUS_KEY] = "error"
        st.session_state[_ERROR_KEY] = str(exc)
        return None


def pipeline_status() -> str:
    """Return 'unloaded' | 'loading' | 'ready' | 'error'."""
    return st.session_state.get(_STATUS_KEY, "unloaded")


def pipeline_error() -> str:
    """Return the last load error message, or empty string."""
    return st.session_state.get(_ERROR_KEY, "")


def is_ready() -> bool:
    return pipeline_status() == "ready"
