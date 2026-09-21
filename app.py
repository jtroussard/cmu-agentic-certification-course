"""
app.py

Mail Organizer Pro — Streamlit GUI

Tabs:
  1. 📥 Upload & Process  — upload a PDF, run the full pipeline, see the result
  2. 🗂️ File Browser      — visual tree of data/organized/ (proposed paths from DB)
  3. 🔍 Manual Review     — approve / correct docs in the NEEDS_HUMAN_REVIEW queue

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Project root on sys.path so gui/ imports work from the repo root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# DEV_MODE: Completely purge all project modules from memory on every rerun
PROJECT_MODULES = ("db", "agents", "gui", "models", "pipeline", "ingestion", "rag", "utils")
for mod in list(sys.modules.keys()):
    if any(mod == p or mod.startswith(p + ".") for p in PROJECT_MODULES):
        sys.modules.pop(mod, None)

from db.database import (
    DEFAULT_DB_PATH,
    approve_review_item,
    get_connection,
    get_directory_tree_nodes,
    get_recent_documents,
    get_review_queue,
    init_db,
    load_directory_taxonomy,
    reject_review_item,
)
from gui.pipeline_manager import (
    get_pipeline,
    is_ready,
    pipeline_error,
    pipeline_status,
)
from gui.log_capture import StreamlitLogCapture
from gui.components.modal import confirm_modal, is_modal_open, open_modal, processing_modal
from gui.components.doc_preview import preview_modal
from agents.verification_agent import VerificationAgent
from rag.vector_store import store_correction, build_descriptor

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Mail Organizer Pro",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Tab bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0f172a;
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: #fff !important;
    }

    /* Cards */
    .mop-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: #f1f5f9;
    }
    .mop-card b { color: #ffffff; }
    .mop-card small { color: #94a3b8; }
    .mop-card code { color: #7dd3fc; background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; }

    /* Result badges */
    .badge-archived  { background:#10b981; color:#fff; padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight:600; }
    .badge-review    { background:#f59e0b; color:#fff; padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight:600; }
    .badge-duplicate { background:#6366f1; color:#fff; padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight:600; }
    .badge-pending   { background:#64748b; color:#fff; padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight:600; }

    /* Tree */
    .tree-dir  { color:#60a5fa; font-weight:600; }
    .tree-file { color:#94a3b8; padding-left:1.5rem; }

    /* Activity item */
    .activity-item { border-left:3px solid #3b82f6; padding-left:0.75rem; margin-bottom:0.6rem; }

    /* Modal Backdrop: dramatic dark tint and frosted blur behind the dialog */
    div[data-testid="stDialog"],
    .stDialog {
        background-color: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(5px) !important;
        -webkit-backdrop-filter: blur(5px) !important;
    }

    /* Modal Card: clean white card with sharp contrast and smooth shadow */
    div[data-testid="stDialog"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 16px !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6) !important;
    }

    /* Ensure text inside modal card is dark and legible */
    div[data-testid="stDialog"] > div * {
        color: #0f172a;
    }
    div[data-testid="stDialog"] > div h1,
    div[data-testid="stDialog"] > div h2,
    div[data-testid="stDialog"] > div h3,
    div[data-testid="stDialog"] > div h4,
    div[data-testid="stDialog"] > div p,
    div[data-testid="stDialog"] > div span,
    div[data-testid="stDialog"] > div label {
        color: #0f172a !important;
    }

    /* Close button on modal */
    div[data-testid="stDialog"] button[aria-label="Close"] svg {
        fill: #475569 !important;
    }

    /* Secondary buttons inside modal */
    div[data-testid="stDialog"] button[kind="secondary"] {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DB_PATH = DEFAULT_DB_PATH


def _badge(decision: str) -> str:
    cls = {
        "ARCHIVED": "badge-archived",
        "NEEDS_HUMAN_REVIEW": "badge-review",
        "DEDUPLICATED": "badge-duplicate",
    }.get(decision, "badge-pending")
    return f'<span class="{cls}">{decision}</span>'


def _tree_from_db() -> dict:
    """
    Build a nested dict representing proposed destination paths from the manifest.
    Uses destination_path stored after ARCHIVED decisions.
    """
    tree: dict = {}
    with get_connection(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT destination_path, new_filename
            FROM document_manifest
            WHERE decision = 'ARCHIVED' AND destination_path IS NOT NULL
            ORDER BY destination_path
            """
        ).fetchall()

    for row in rows:
        parts = Path(row["destination_path"]).parts
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
        # leaf — store filename
        filename = row["new_filename"] or ""
        node.setdefault("__files__", []).append(filename)

    return tree


def _render_tree(node: dict, indent: int = 0) -> None:
    """Recursively render the directory tree."""
    for key, child in node.items():
        if key == "__files__":
            for f in child:
                st.markdown(
                    f'<div class="tree-file">{"&nbsp;" * (indent * 4)}📄 {f}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div class="tree-dir">{"&nbsp;" * (indent * 4)}📁 {key}</div>',
                unsafe_allow_html=True,
            )
            _render_tree(child, indent + 1)


def _status_color(status: str) -> str:
    return {"ready": "🟢", "loading": "🟡", "error": "🔴", "unloaded": "⚪"}.get(
        status, "⚪"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📬 Mail Organizer Pro")
    st.markdown("---")

    # Pipeline status
    status = pipeline_status()
    st.markdown(f"**Pipeline** {_status_color(status)} `{status.upper()}`")

    if status == "error":
        st.error(pipeline_error())

    if status == "unloaded":
        st.info("Pipeline loads automatically on first upload.")

    if status == "ready":
        if st.button("🔄 Reload Models", use_container_width=True):
            for key in ["_mop_pipeline", "_mop_status", "_mop_error"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.markdown("---")

    # Recent activity
    st.markdown("**Recent Activity**")
    try:
        init_db(DB_PATH)
        with get_connection(DB_PATH) as conn:
            recent = get_recent_documents(conn, limit=5)
        if recent:
            for doc in recent:
                decision = doc["decision"]
                icon = {"ARCHIVED": "✅", "NEEDS_HUMAN_REVIEW": "⚠️"}.get(decision, "📄")
                st.markdown(
                    f'<div class="activity-item">'
                    f"{icon} <b>{doc['original_filename']}</b><br>"
                    f'<small style="color:#64748b">{decision} · {doc["processed_at"][:10]}</small>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No documents processed yet.")
    except Exception:
        st.caption("DB not initialised yet.")

    st.markdown("---")
    st.caption("CMU Agentic Certificate Program")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_upload, tab_browser, tab_review = st.tabs(
    ["📥 Upload & Process", "🗂️ File Browser", "🔍 Manual Review"]
)

# ===========================================================================
# TAB 1 — Upload & Process
# ===========================================================================

with tab_upload:
    st.markdown("### Upload a Document")
    st.markdown(
        "Drop a PDF below. The agentic pipeline will extract metadata, "
        "verify routing, and either archive the document or flag it for review."
    )

    uploader_key = f"pdf_uploader_{st.session_state.get('uploader_version', 0)}"
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key=uploader_key,
    )

    if uploaded_file is not None:
        col_info, col_run = st.columns([3, 1])
        with col_info:
            st.markdown(
                f'<div class="mop-card">📄 <b>{uploaded_file.name}</b>'
                f"<br><small>Size: {uploaded_file.size / 1024:.1f} KB</small></div>",
                unsafe_allow_html=True,
            )
        with col_run:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

        if run_btn:
            # Save upload to a temp location inside the project
            tmp_dir = ROOT / "data" / "_uploads"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / uploaded_file.name
            tmp_path.write_bytes(uploaded_file.getvalue())

            with st.spinner("Loading models & processing … this may take a minute."):
                pipeline = get_pipeline(db_path=DB_PATH)

            if pipeline is None:
                st.error(f"❌ Failed to load pipeline: {pipeline_error()}")
            else:
                status_placeholder = st.empty()
                status_placeholder.markdown(
                    "<p style='color:#94a3b8; font-size:0.85rem; margin:0.25rem 0 0;'>"
                    "⚙️ Running extraction + verification …</p>",
                    unsafe_allow_html=True,
                )
                log_placeholder = st.empty()
                try:
                    with StreamlitLogCapture(log_placeholder):
                        result = pipeline.process(tmp_path)
                    vr = result.verification_result
                    st.session_state["last_result"] = (result, uploaded_file.name)
                    status_placeholder.markdown(
                        "<p style='color:#10b981; font-size:0.85rem; margin:0.25rem 0 0;'>"
                        "✅ Pipeline run complete</p>",
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    status_placeholder.markdown(
                        "<p style='color:#ef4444; font-size:0.85rem; margin:0.25rem 0 0;'>"
                        "❌ Pipeline error</p>",
                        unsafe_allow_html=True,
                    )
                    st.error(f"❌ Pipeline error: {exc}")
                    st.session_state.pop("last_result", None)

        # Display last result
        if "last_result" in st.session_state:
            result, fname = st.session_state["last_result"]
            vr = result.verification_result

            st.markdown("---")
            st.markdown("#### Result")

            if result.deduplicated:
                st.markdown(
                    '<div class="mop-card">'
                    f'{_badge("DEDUPLICATED")} &nbsp; <b>{fname}</b><br>'
                    "<small>Already in manifest — skipped.</small>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                decision = vr.decision.value
                col1, col2 = st.columns(2)
                col1.metric("Decision", decision)
                col2.metric("Confidence Gate", "✅ Pass" if vr.confidence_check_passed else "❌ Fail")

                st.markdown(
                    '<div class="mop-card">'
                    f"<b>Destination:</b> <code>{vr.destination_path or 'N/A'}</code><br>"
                    f"<b>Filename:</b> <code>{vr.new_filename or 'N/A'}</code><br>"
                    f"<b>New Dir:</b> {'Yes ⚠️' if vr.requires_new_directory else 'No'}<br>"
                    f"<b>Reason:</b> {vr.reason}"
                    "</div>",
                    unsafe_allow_html=True,
                )

                if decision == "NEEDS_HUMAN_REVIEW":
                    st.warning("This document has been added to the Manual Review queue.")
                elif decision == "ARCHIVED":
                    st.success("Document routed successfully.")

            if st.button("Clear Result"):
                st.session_state.pop("last_result", None)
                st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1
                st.rerun()

# ===========================================================================
# TAB 2 — File Browser
# ===========================================================================

with tab_browser:
    st.markdown("### Organised File Tree")
    st.markdown(
        "Displays the proposed destination paths for all **archived** documents. "
        "Refresh to see newly processed files."
    )

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Tree"):
            st.rerun()

    try:
        init_db(DB_PATH)
        tree = _tree_from_db()
        if tree:
            st.markdown('<div class="mop-card">', unsafe_allow_html=True)
            _render_tree(tree)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No archived documents yet. Upload and process a PDF to get started.")

        # Stats row
        with get_connection(DB_PATH) as conn:
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN decision='ARCHIVED' THEN 1 ELSE 0 END) as archived,
                    SUM(CASE WHEN decision='NEEDS_HUMAN_REVIEW' THEN 1 ELSE 0 END) as review,
                    SUM(CASE WHEN decision='DEDUPLICATED' THEN 1 ELSE 0 END) as dupes
                FROM document_manifest
                """
            ).fetchone()

        if stats and stats["total"]:
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Processed", stats["total"])
            c2.metric("Archived", stats["archived"])
            c3.metric("Needs Review", stats["review"])
            c4.metric("Duplicates", stats["dupes"])

    except Exception as exc:
        st.error(f"Error loading tree: {exc}")

# ===========================================================================
# TAB 3 — Manual Review
# ===========================================================================

with tab_review:
    try:
        init_db(DB_PATH)
        with get_connection(DB_PATH) as conn:
            queue = get_review_queue(conn)

        st.markdown("### Manual Review Queue")
        st.markdown(
            "Documents that failed confidence gating or require a new directory. "
            "Edit the fields and click **Approve** to archive."
        )
        if not queue:
            st.success("✅ Review queue is empty — all documents are accounted for.")
        else:
            st.markdown(f"**{len(queue)} document(s) pending review**")

        def _render_item_guidance(item_data: dict, parsed_ext: dict) -> None:
            """Render the AI assistant guidance card for a specific queue item (SRP/DRY)."""
            ai_msg = parsed_ext.get("_ai_summary")
            if not ai_msg:
                doc_name = item_data.get("original_filename", "the document")
                dest_missing = not item_data.get("proposed_path")
                conf = item_data.get("overall_confidence", 0.0)
                if dest_missing and conf >= 0.8:
                    ai_msg = (
                        f"We successfully identified the key details for **{doc_name}** ({conf:.2f} confidence), "
                        "but need you to choose a **Destination Path** before it can be archived."
                    )
                else:
                    ai_msg = (
                        f"**{doc_name}** needs human review: please verify any highlighted fields below "
                        "and enter a destination folder to complete archiving."
                    )

            st.markdown(
                f"""
                <div style="background: rgba(59, 130, 246, 0.08); border-left: 4px solid #3b82f6; border-radius: 6px; padding: 12px 16px;">
                    <div style="font-weight: 600; color: #1d4ed8; margin-bottom: 4px; font-size: 0.92rem;">
                        🤖 Assistant Guidance
                    </div>
                    <div style="font-size: 0.88rem; color: #1e293b; line-height: 1.45;">
                        {ai_msg}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if queue:
            for item in queue:
                with st.expander(
                    f"📄 {item['original_filename']}  |  confidence: {item['overall_confidence']:.2f}",
                    expanded=True,
                ):
                    # Parse stored extraction_json if present
                    ext_data: dict = {}
                    if item.get("extraction_json"):
                        try:
                            ext_data = json.loads(item["extraction_json"])
                        except Exception:
                            ext_data = {}

                    # Top section: Details on left, Assistant Guidance on right
                    top_left, top_right = st.columns([1.1, 1.0])
                    with top_left:
                        st.markdown(f"**Reason:** {item['reason']}")
                        st.markdown(
                            f"**Queued:** {item['queued_at'][:19]}  |  "
                            f"**Requires new dir:** {'Yes' if item['requires_new_dir'] else 'No'}"
                        )
                        st.markdown(f"**Source:** `{item['source_path']}`")
                        if st.button("👁️ Preview Document", key=f"preview_btn_{item['id']}"):
                            preview_modal(item["source_path"], item["original_filename"])
                    with top_right:
                        _render_item_guidance(item, ext_data)

                    def _get_val(field_key: str) -> str:
                        f = ext_data.get(field_key, {})
                        if isinstance(f, dict):
                            c = float(f.get("confidence", 0.0))
                            v = f.get("value")
                            if c > 0.0 and v is not None:
                                return str(v)
                        return ""

                    def _get_conf(field_key: str) -> float:
                        f = ext_data.get(field_key, {})
                        if isinstance(f, dict):
                            return float(f.get("confidence", 0.0))
                        return 0.0

                    def _field_label(label: str, field_key: str) -> str:
                        conf = _get_conf(field_key)
                        val = _get_val(field_key)
                        curr_doc_type = (_get_val("document_type") or "").strip().lower()
                        is_acct_type = any(k in curr_doc_type for k in ("utility", "bill", "banking", "finance", "mortgage", "tax"))

                        if field_key == "document_identifier":
                            if val and conf >= 0.85:
                                return f"✅ {label} ({conf:.2f})"
                            elif val:
                                return f"🟡 {label} ({conf:.2f} — please confirm)"
                            else:
                                return f"⚪ {label} (optional)"

                        if field_key == "account_number":
                            if val and val.lower() not in ("no-account-number", "no-account", "none"):
                                if conf >= 0.85:
                                    return f"✅ {label} ({conf:.2f})"
                                else:
                                    return f"🟡 {label} ({conf:.2f} — please confirm)"
                            elif val and val.lower() == "no-account-number":
                                return f"⚪ {label} (no account required)"
                            elif not is_acct_type:
                                return f"⚪ {label} (optional / one-off)"
                            else:
                                return f"🔴 {label} (missing / needs input)"

                        if not val or conf == 0.0:
                            return f"🔴 {label} (missing / needs input)"
                        elif conf < 0.85:
                            return f"🟡 {label} ({conf:.2f} — please confirm)"
                        else:
                            return f"✅ {label} ({conf:.2f})"

                    st.markdown("##### Extracted Metadata")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        meta_entity = st.text_input(
                            _field_label("Entity / Vendor", "entity"),
                            value=_get_val("entity"),
                            key=f"entity_{item['id']}",
                            placeholder="e.g. Apex Plumbing",
                        )
                        meta_doc_type = st.text_input(
                            _field_label("Document Type", "document_type"),
                            value=_get_val("document_type"),
                            key=f"type_{item['id']}",
                            placeholder="e.g. invoice",
                        )
                    with m_col2:
                        meta_date = st.text_input(
                            _field_label("Document Date", "document_date"),
                            value=_get_val("document_date"),
                            key=f"date_{item['id']}",
                            placeholder="YYYY-MM-DD",
                        )
                        meta_account = st.text_input(
                            _field_label("Account / ID #", "account_number"),
                            value=_get_val("account_number"),
                            key=f"acct_{item['id']}",
                            placeholder="e.g. 767",
                        )
                    with m_col3:
                        meta_amount = st.text_input(
                            _field_label("Total Amount", "total_amount"),
                            value=_get_val("total_amount"),
                            key=f"amount_{item['id']}",
                            placeholder="e.g. 150.00",
                        )
                        meta_doc_id = st.text_input(
                            _field_label("Document ID", "document_identifier"),
                            value=_get_val("document_identifier"),
                            key=f"doc_id_{item['id']}",
                            placeholder="e.g. Invoice #770, RO-4412, Docket 2024-CR-1187",
                        )

                    st.markdown("##### Archive Destination")

                    # Fetch tree nodes and canonical taxonomy
                    with get_connection(DB_PATH) as conn:
                        tree_nodes = get_directory_tree_nodes(conn)
                    taxonomy = load_directory_taxonomy()

                    # Parse initial path suggestion if present
                    prop_path = (item.get("proposed_path") or "").strip().strip("/")
                    prop_parts = prop_path.split("/") if prop_path else []

                    init_l1 = prop_parts[0] if (prop_parts and prop_parts[0] in tree_nodes) else ""
                    init_l2 = prop_parts[1] if len(prop_parts) > 1 else ""
                    init_l3 = "/".join(prop_parts[2:]) if len(prop_parts) > 2 else ""

                    # Level 1 — Domain
                    l1_placeholder = "-- Select Domain --"
                    l1_options_raw = list(tree_nodes.keys())
                    if init_l1 and init_l1 in l1_options_raw:
                        l1_options = l1_options_raw
                        l1_idx = l1_options_raw.index(init_l1)
                    else:
                        l1_options = [l1_placeholder] + l1_options_raw
                        l1_idx = 0
                    curr_l1 = st.session_state.get(f"l1_{item['id']}", init_l1 or l1_placeholder)
                    l1_desc = taxonomy.get(curr_l1, {}).get("description", "Canonical root directory domain (fixed).")

                    c_l1, c_l2, c_l3 = st.columns(3)
                    with c_l1:
                        sel_l1 = st.selectbox(
                            "Level 1: Domain",
                            options=l1_options,
                            index=l1_idx,
                            key=f"l1_{item['id']}",
                            help=f"{curr_l1}: {l1_desc}",
                        )

                    # Level 2 — Entity / Property / Account
                    existing_entities = sorted(list(tree_nodes.get(sel_l1, {}).keys()))
                    l2_new_label = "➕ [New Entity / Property...]"
                    l2_select_prompt = "-- Select Entity / Property --"

                    if init_l2 and init_l2 in existing_entities:
                        l2_options = existing_entities + [l2_new_label]
                        l2_default_idx = existing_entities.index(init_l2)
                    elif init_l2 and init_l2 not in existing_entities:
                        l2_options = existing_entities + [l2_new_label]
                        l2_default_idx = len(l2_options) - 1
                    else:
                        l2_options = [l2_select_prompt] + existing_entities + [l2_new_label]
                        l2_default_idx = 0

                    entity_rule = taxonomy.get(sel_l1, {}).get("entity_rule", "Entity identifier")
                    entity_tip = taxonomy.get(sel_l1, {}).get("entity_tooltip", "Provide entity name with underscores, no spaces.")

                    with c_l2:
                        sel_l2_choice = st.selectbox(
                            "Level 2: Entity / Property",
                            options=l2_options,
                            index=l2_default_idx,
                            key=f"l2_{item['id']}",
                            help=f"Rule for {sel_l1}: {entity_rule}. {entity_tip}",
                        )
                        if sel_l2_choice == l2_new_label:
                            new_entity_val = st.text_input(
                                "Enter New Entity",
                                value=init_l2 if (init_l2 and init_l2 not in existing_entities) else "",
                                key=f"l2_new_{item['id']}",
                                placeholder=f"e.g. {entity_rule}",
                                help=entity_tip,
                            )
                            sel_l2 = new_entity_val.strip().replace(" ", "_")
                        elif sel_l2_choice == l2_select_prompt:
                            sel_l2 = ""
                        else:
                            sel_l2 = sel_l2_choice

                    # Level 3 — Category
                    std_cats = taxonomy.get(sel_l1, {}).get("default_categories", [])
                    tree_cats = tree_nodes.get(sel_l1, {}).get(sel_l2, [])
                    all_cats = sorted(list(set(std_cats + tree_cats)))
                    l3_new_label = "➕ [New Category...]"
                    l3_select_prompt = "-- Select Category --"

                    if init_l3 and init_l3 in all_cats:
                        l3_options = all_cats + [l3_new_label]
                        l3_default_idx = all_cats.index(init_l3)
                    elif init_l3 and init_l3 not in all_cats:
                        l3_options = all_cats + [l3_new_label]
                        l3_default_idx = len(l3_options) - 1
                    else:
                        l3_options = [l3_select_prompt] + all_cats + [l3_new_label]
                        l3_default_idx = 0

                    cat_tip = taxonomy.get(sel_l1, {}).get("category_tooltip", "Select document subcategory.")

                    with c_l3:
                        sel_l3_choice = st.selectbox(
                            "Level 3: Category",
                            options=l3_options,
                            index=l3_default_idx,
                            key=f"l3_{item['id']}",
                            help=f"Category for {sel_l1}: {cat_tip}",
                        )
                        if sel_l3_choice == l3_new_label:
                            new_cat_val = st.text_input(
                                "Enter New Category",
                                value=init_l3 if (init_l3 and init_l3 not in all_cats) else "",
                                key=f"l3_new_{item['id']}",
                                placeholder="e.g. Repairs",
                                help="Keep concise, no spaces (use underscores).",
                            )
                            sel_l3 = new_cat_val.strip().replace(" ", "_")
                        elif sel_l3_choice == l3_select_prompt:
                            sel_l3 = ""
                        else:
                            sel_l3 = sel_l3_choice

                    st.caption("💡 To add a new entity or category not listed in the dropdown, choose **➕ [New...]**.")

                    # Assembled Canonical Path & Breadcrumb Display
                    path_parts = [p for p in [sel_l1, sel_l2, sel_l3] if p]
                    dest = "/".join(path_parts)

                    col_bread, col_name = st.columns(2)
                    with col_bread:
                        breadcrumb_html = (
                            f'<div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 14px; margin-top: 4px;">'
                            f'<div style="font-size: 0.78rem; color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 2px;">📁 Destination Breadcrumb</div>'
                            f'<div style="font-family: monospace; font-size: 0.92rem; font-weight: 600; color: #1e293b;">'
                            f'<span style="color:#2563eb;">{sel_l1}</span> › '
                            f'<span style="color:#0d9488;">{sel_l2 or "⚠️ [select entity]"}</span> › '
                            f'<span style="color:#475569;">{sel_l3 or "⚠️ [select category]"}</span>'
                            f'</div>'
                            f'</div>'
                        )
                        st.markdown(breadcrumb_html, unsafe_allow_html=True)
                    with col_name:
                        suggested_name = item["proposed_filename"] or "(Generated automatically by agent upon approval)"
                        st.text_input(
                            "New Filename (Auto-generated by Agent)",
                            value=suggested_name,
                            key=f"fname_{item['id']}",
                            disabled=True,
                            help="Canonical 4-part filename is synthesized automatically by the Verification Agent upon approval.",
                        )

                    col_approve, col_reject, _ = st.columns([1, 1, 4])
                    approve_loader_key = f"loader_approve_{item['id']}"
                    with col_approve:
                        if st.button("✅ Approve", key=f"approve_{item['id']}", type="primary"):
                            if not sel_l2 or not sel_l3:
                                st.error("Both an Entity/Property (Level 2) and Category (Level 3) must be selected or entered.")
                            else:
                                st.session_state[f"_staged_approve_{item['id']}"] = {
                                    "document_id": item["document_id"],
                                    "document_type": meta_doc_type,
                                    "entity": meta_entity,
                                    "description": meta_doc_type,
                                    "document_date": meta_date,
                                    "account_number": meta_account or None,
                                    "document_identifier": meta_doc_id or None,
                                    "total_amount": float(meta_amount) if meta_amount and meta_amount.replace(".", "", 1).isdigit() else None,
                                    "file_path": item.get("source_path") or item.get("original_filename") or "document.pdf",
                                    "destination_path": dest,
                                    "extraction_json": item.get("extraction_json"),
                                    "source_path": item.get("source_path"),
                                }
                                open_modal(approve_loader_key)
                                st.rerun()

                    # Render approval processing modal with spinner and darkened backdrop
                    if is_modal_open(approve_loader_key):
                        staged = st.session_state.get(f"_staged_approve_{item['id']}")
                        if staged:
                            def _run_approval_work():
                                time.sleep(0.4)
                                now = datetime.now(timezone.utc).isoformat()
                                with get_connection(DB_PATH) as conn:
                                    confirmed_payload = {
                                        "document_id": staged["document_id"],
                                        "document_type": staged["document_type"],
                                        "entity": staged["entity"],
                                        "description": staged["description"],
                                        "document_date": staged["document_date"],
                                        "account_number": staged["account_number"],
                                        "document_identifier": staged.get("document_identifier"),
                                        "total_amount": staged["total_amount"],
                                        "file_path": staged["file_path"],
                                    }
                                    verifier = VerificationAgent(model=lambda x: {})
                                    canonical_dest, canonical_fname = verifier.re_verify(
                                        payload=confirmed_payload,
                                        destination_path=staged["destination_path"],
                                        conn=conn,
                                    )
                                    approve_review_item(
                                        conn=conn,
                                        queue_id=item["id"],
                                        document_id=staged["document_id"],
                                        destination_path=canonical_dest,
                                        new_filename=canonical_fname,
                                        resolved_at=now,
                                    )

                                # Save human correction into RAG vector store for self-learning (outside conn context to prevent lock collision)
                                try:
                                    from models.llm_adapter import build_embedding_fn
                                    embed_fn = build_embedding_fn()

                                    # Build normalized descriptor from human-approved fields
                                    # (preferred over raw OCR — stable, vendor-specific signal)
                                    descriptor = build_descriptor(
                                        entity=staged.get("entity", "") or "",
                                        document_type=staged.get("document_type", "") or "",
                                        account_number=staged.get("account_number", "") or "",
                                        correct_path=canonical_dest,
                                        document_identifier=staged.get("document_identifier", "") or "",
                                    )

                                    # Use descriptor as the text_slice (also used for dedup hash)
                                    # Fall back to raw OCR slice only if descriptor is degenerate
                                    slice_val = descriptor if descriptor else ""
                                    if not slice_val:
                                        if staged.get("extraction_json"):
                                             try:
                                                 ej = json.loads(staged["extraction_json"])
                                                 slice_val = ej.get("extracted_text_slice", "")
                                             except Exception:
                                                 pass
                                        if not slice_val and staged.get("source_path"):
                                            try:
                                                from ingestion.parser import extract_text_from_pdf
                                                from ingestion.slicer import slice_text
                                                p_src = Path(staged["source_path"])
                                                if not p_src.exists():
                                                    p_src = ROOT / staged["source_path"]
                                                if p_src.exists():
                                                    slice_val = slice_text(extract_text_from_pdf(p_src))
                                            except Exception:
                                                pass

                                    if slice_val:
                                        rec_id = store_correction(
                                            text_slice=slice_val,
                                            entity=staged["entity"],
                                            account_number=staged["account_number"] or "",
                                            document_type=staged["document_type"],
                                            correct_path=canonical_dest,
                                            embed_fn=embed_fn,
                                            db_path=DB_PATH,
                                            document_identifier=staged.get("document_identifier", "") or "",
                                        )
                                        print(f"[RAG Store] Saved correction #{rec_id} descriptor='{descriptor}' -> {canonical_dest}")
                                    else:
                                        print("[RAG Store Error] No text slice or descriptor available to embed.")
                                except Exception as e_rag:
                                    print(f"[RAG Store Error] {e_rag}")
                                st.session_state.pop("last_result", None)
                                st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1
                                st.success(f"✅ Approved and archived: {canonical_fname}")
                                time.sleep(0.7)
                                st.rerun()

                            processing_modal(
                                key=approve_loader_key,
                                title="Agent Review & Archival",
                                message="Agent reviewing updates and archiving document...",
                                work_fn=_run_approval_work,
                            )
                    with col_reject:
                        modal_key = f"reject_{item['id']}"
                        if st.button("❌ Reject", key=f"reject_btn_{item['id']}"):
                            open_modal(modal_key)
                            st.rerun()
                        if confirm_modal(
                            title="Reject Document?",
                            body=(
                                "⚠️ Rejecting will **permanently remove** this document from "
                                "the dedup manifest so it can be re-uploaded and processed fresh. "
                                "The audit log entry will be preserved."
                            ),
                            confirm_label="Yes, reject & purge",
                            cancel_label="Cancel",
                            danger=True,
                            key=modal_key,
                        ):
                            now = datetime.now(timezone.utc).isoformat()
                            with get_connection(DB_PATH) as conn:
                                reject_review_item(
                                    conn=conn,
                                    queue_id=item["id"],
                                    document_id=item["document_id"],
                                    resolved_at=now,
                                )
                            st.session_state.pop("last_result", None)
                            st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1
                            st.rerun()

    except Exception as exc:
        st.error(f"Error loading review queue: {exc}")
