"""
gui/log_capture.py

Context manager that intercepts stdout (which the agents use for print() output)
and streams each line into a Streamlit st.empty() placeholder as a terminal-style
log tile.

Usage:
    log_box = st.empty()
    with StreamlitLogCapture(log_box):
        pipeline.process(path)
"""

from __future__ import annotations

import sys
from io import StringIO
from datetime import datetime


class StreamlitLogCapture:
    """
    Redirect sys.stdout so that every print() line from the pipeline/agents
    is appended to an in-memory buffer and re-rendered into the given
    Streamlit placeholder in real time.
    """

    MAX_LINES = 80  # cap so the tile doesn't grow unbounded

    def __init__(self, placeholder, title: str = "⚙️ Agent Log") -> None:
        self._placeholder = placeholder
        self._title = title
        self._lines: list[str] = []
        self._original_stdout = sys.stdout
        self._buffer = _WriteThroughBuffer(self._on_write, self._original_stdout)

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "StreamlitLogCapture":
        sys.stdout = self._buffer
        self._render()
        return self

    def __exit__(self, *_) -> None:
        sys.stdout = self._original_stdout
        # Final render showing completion state
        self._render_complete()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_write(self, text: str) -> None:
        """Called by the buffer for every write() call to stdout."""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                ts = datetime.now().strftime("%H:%M:%S")
                self._lines.append(f"[{ts}] {stripped}")
        # Keep the tail only
        if len(self._lines) > self.MAX_LINES:
            self._lines = self._lines[-self.MAX_LINES :]
        self._render()

    def _render(self) -> None:
        self._render_tile(title=self._title, border_color="#1e3a5f", title_color="#38bdf8")

    def _render_complete(self) -> None:
        self._render_tile(
            title="✅ Pipeline run complete",
            border_color="#064e3b",
            title_color="#10b981",
            footer="— Agent finished —",
        )

    def _render_tile(
        self,
        title: str,
        border_color: str,
        title_color: str,
        footer: str | None = None,
    ) -> None:
        body = "\n".join(self._lines) if self._lines else "Waiting for output…"
        footer_html = (
            f"<div style='color:#10b981; margin-top:0.5rem; font-size:0.75rem;'>{_escape(footer)}</div>"
            if footer
            else ""
        )
        self._placeholder.markdown(
            f"""
<div style="
    background: #020617;
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 0.78rem;
    line-height: 1.6;
    color: #94a3b8;
    max-height: 320px;
    overflow-y: auto;
    margin-top: 0.75rem;
">
<div style="color:{title_color}; font-weight:600; margin-bottom:0.5rem; font-size:0.8rem;">
    {title}
</div>
<div style="white-space: pre-wrap; word-break: break-all;">{_escape(body)}</div>
{footer_html}
</div>
""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _WriteThroughBuffer:
    """
    A file-like object that calls a callback on each write AND also passes
    the data through to the original stdout so the terminal still works.
    """

    def __init__(self, callback, passthrough) -> None:
        self._callback = callback
        self._passthrough = passthrough

    def write(self, text: str) -> int:
        if text:
            self._callback(text)
            self._passthrough.write(text)
        return len(text)

    def flush(self) -> None:
        self._passthrough.flush()

    def isatty(self) -> bool:
        return False


def _escape(text: str) -> str:
    """Minimal HTML escaping for safe insertion into a div."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )
