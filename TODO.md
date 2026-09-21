# TODO: Markdown-to-HTML Text Converter for Assistant Guidance

## Objective
Fix raw markdown formatting (such as `**bold**` asterisks) appearing literally inside the HTML Assistant Guidance card in the Manual Review queue.

## Proposed Solution: Simple Converter Function
Create a lightweight converter utility (e.g. in `gui/components/formatters.py` or directly in `app.py`) that converts standard markdown inline tokens into valid HTML tags before injecting into raw HTML templates:

```python
import re

def markdown_to_html(text: str) -> str:
    """Convert common inline markdown tags to HTML tags."""
    if not text:
        return ""
    # Bold: **text** or __text__ -> <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    # Italic: *text* or _text_ -> <i>text</i>
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    # Inline code: `code` -> <code>code</code>
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text
```

## Integration Point
In [`app.py`](file:///Users/jtroussard/school/cmu/agentic-certificate-program/repos/mail-organizer-pro/app.py) inside the Manual Review tab:
```python
formatted_msg = markdown_to_html(ai_msg)
st.markdown(
    f"""
    <div style="...">
        ...
        <div>{formatted_msg}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
```

## Verification
- Refresh `http://localhost:8501`.
- Verify that `**plumbing-repair_767_45-craighead.pdf**` renders as bold text without literal asterisks.

---

# TODO: Turn All Caches Back On (Post-Launch / Production)

## Objective
Re-enable caching mechanisms across the application once development is 100% complete, restoring high performance and avoiding redundant disk / model reloads.

## Checklist
1. **Remove `sys.modules` purge in `app.py`**:
   Remove the startup loop that pops `PROJECT_MODULES` from `sys.modules`.
2. **Re-enable Pipeline Caching**:
   In [`gui/pipeline_manager.py`](file:///Users/jtroussard/school/cmu/agentic-certificate-program/repos/mail-organizer-pro/gui/pipeline_manager.py), set `DEV_MODE = False` so heavy models and pipeline instances persist across runs in `st.session_state`.
3. **Re-enable Python Bytecode**:
   Remove `export PYTHONDONTWRITEBYTECODE=1` from startup scripts so Python can compile and reuse `.pyc` files.
4. **Streamlit Production Settings**:
   In [`.streamlit/config.toml`](file:///Users/jtroussard/school/cmu/agentic-certificate-program/repos/mail-organizer-pro/.streamlit/config.toml), change `fileWatcherType = "auto"` or disable file polling for production deployments.

