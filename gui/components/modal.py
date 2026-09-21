"""
gui/components/modal.py

Generic confirm modal using Streamlit's @st.dialog decorator (requires
Streamlit ≥ 1.36).  Wrap any confirm/alert pattern with confirm_modal().

Usage:
    from gui.components.modal import confirm_modal

    if confirm_modal(
        title="Are you sure?",
        body="This action cannot be undone.",
        confirm_label="Yes, proceed",
        danger=True,
        key="my_action",
    ):
        do_the_thing()
        st.rerun()

How it works:
  - On first call the dialog is shown.
  - Returns True only once — when the user clicks the confirm button inside
    the dialog.  The caller should call st.rerun() after acting so the dialog
    clears itself from session_state.
"""

import streamlit as st


def inject_modal_backdrop_css() -> None:
    """Inject CSS to dramatically darken and blur background behind all modals, and keep the modal card crisp white with dark readable text."""
    st.markdown(
        """
        <style>
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


def confirm_modal(
    title: str,
    body: str,
    confirm_label: str = "Confirm",
    cancel_label: str = "Cancel",
    danger: bool = False,
    key: str = "confirm_modal",
) -> bool:
    """
    Show a modal dialog asking the user to confirm an action.
    """
    trigger_key = f"_modal_open_{key}"
    result_key = f"_modal_confirmed_{key}"

    # If the confirmed flag is set, clear it and return True to the caller
    if st.session_state.get(result_key):
        st.session_state.pop(result_key, None)
        return True

    # Show trigger button that opens the dialog
    if not st.session_state.get(trigger_key):
        return False  # dialog not open — caller renders its own open button

    # Dialog is open — render it
    @st.dialog(title)
    def _dialog() -> None:
        inject_modal_backdrop_css()
        st.write(body)
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            btn_type = "primary" if danger else "secondary"
            if st.button(confirm_label, type=btn_type, key=f"{key}_confirm_btn"):
                st.session_state.pop(trigger_key, None)
                st.session_state[result_key] = True
                st.rerun()
        with col_cancel:
            if st.button(cancel_label, key=f"{key}_cancel_btn"):
                st.session_state.pop(trigger_key, None)
                st.rerun()

    _dialog()
    return False


def open_modal(key: str) -> None:
    """Set the session_state flag that causes a modal to open."""
    st.session_state[f"_modal_open_{key}"] = True


def is_modal_open(key: str) -> bool:
    """Check if the modal open flag is set for a key."""
    return bool(st.session_state.get(f"_modal_open_{key}"))


def close_modal(key: str) -> None:
    """Clear the session_state flag to close the modal."""
    st.session_state.pop(f"_modal_open_{key}", None)


def processing_modal(
    key: str,
    title: str,
    message: str,
    work_fn,
) -> None:
    """
    Generic loader/spinner modal extending the base modal architecture.
    Opens with a darkened background and a spinner message while work_fn executes.
    """
    if not is_modal_open(key):
        return

    @st.dialog(title, width="small")
    def _loader_dialog() -> None:
        inject_modal_backdrop_css()
        st.markdown(f"#### {title}")
        with st.spinner(message):
            try:
                work_fn()
            finally:
                close_modal(key)

    _loader_dialog()
