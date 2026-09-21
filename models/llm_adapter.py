"""
models/llm_adapter.py

Adapters that wrap external model libraries into the LLMCallable and
EmbeddingCallable interfaces defined by this project.

LLMCallable:       Callable[[list[dict]], dict]
EmbeddingCallable: Callable[[str], list[float]]

Two separate builders are provided because the ExtractionAgent needs
native function-calling (tool_calls) while the VerificationAgent only
needs plain JSON text output.

Dependencies (not bundled — install separately, see README):
  llama-cpp-python with Metal:
    CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
  sentence-transformers:
    pip install sentence-transformers
  huggingface_hub:
    pip install huggingface_hub
"""

import json
import uuid
from pathlib import Path
from typing import Any

from agents.extraction_agent import TOOL_SPECS
from .config import VERBOSE_MODE


# ---------------------------------------------------------------------------
# TOOL_SPECS → OpenAI function-calling schema
# ---------------------------------------------------------------------------

def _tool_specs_to_openai(specs: list[dict]) -> list[dict]:
    """
    Convert our internal TOOL_SPECS format to the OpenAI-compatible schema
    that llama-cpp-python expects for native function calling.

    Input spec parameter values are "type — description" strings.
    Output: standard JSON Schema object format.
    """
    tools = []
    for spec in specs:
        properties = {}
        for param_name, param_desc in spec.get("parameters", {}).items():
            # Strip the "string — " type prefix if present
            desc = str(param_desc)
            if " — " in desc:
                desc = desc.split(" — ", 1)[1]
            properties[param_name] = {"type": "string", "description": desc}

        tools.append({
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                },
            },
        })
    return tools


# ---------------------------------------------------------------------------
# Transcript format conversion
# ---------------------------------------------------------------------------

def _to_llama_transcript(transcript: list[dict]) -> list[dict]:
    """
    Convert our internal transcript format to llama-cpp-python's expected format.

    Our format:
      assistant: {"role": "assistant", "content": str, "tool_calls": [{"name": ..., "args": {...}}]}
      tool:      {"role": "tool", "name": str, "content": str}

    llama-cpp format:
      assistant: {"role": "assistant", "content": None, "tool_calls": [
                    {"id": str, "type": "function", "function": {"name": str, "arguments": json_str}}
                  ]}
      tool:      {"role": "tool", "tool_call_id": str, "content": str}
    """
    converted = []
    # Map tool names to their generated call IDs so tool result messages match
    call_id_map: dict[str, str] = {}

    for msg in transcript:
        role = msg.get("role", "")

        if role == "assistant":
            raw_calls = msg.get("tool_calls", [])
            if raw_calls:
                lc_calls = []
                for call in raw_calls:
                    call_id = str(uuid.uuid4())[:8]
                    call_id_map[call.get("name", "")] = call_id
                    lc_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call.get("name", ""),
                            "arguments": json.dumps(call.get("args", {})),
                        },
                    })
                converted.append({
                    "role": "assistant",
                    "content": msg.get("content") or None,
                    "tool_calls": lc_calls,
                })
            else:
                converted.append({"role": "assistant", "content": msg.get("content", "")})

        elif role == "tool":
            tool_name = msg.get("name", "")
            call_id = call_id_map.get(tool_name, "unknown")
            converted.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": msg.get("content", ""),
            })

        else:
            # system / user — pass through unchanged
            converted.append({"role": role, "content": msg.get("content", "")})

    return converted


def _from_llama_response(response: dict) -> dict:
    """
    Convert a llama-cpp-python chat completion response to our internal format.

    Handles:
      1. Standard OpenAI-style message['tool_calls']
      2. Qwen native XML <tool_call> tags emitted in message['content']
    """
    choice = response["choices"][0]["message"]
    raw_content = choice.get("content") or ""
    raw_calls = choice.get("tool_calls") or []

    # Log raw text for full debugging visibility
    if raw_content:
        print(f"\n[llm_adapter] Raw LLM output:\n{raw_content}")

    tool_calls = []
    for tc in raw_calls:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        # Ensure redundant 'text' is never stored in tool arguments
        args.pop("text", None)
        tool_calls.append({"name": fn.get("name", ""), "args": args})

    text = raw_content
    # Fallback: Parse Qwen <tool_call> XML tags if tool_calls list is empty
    if not tool_calls and "<tool_call>" in text:
        import re

        def _lenient_loads(s: str) -> dict | None:
            """
            Multi-strategy JSON parser to handle LLM output quirks:
              1. Direct JSON decoding
              2. Double-brace {{ and }} stripping
              3. Escape repair for invalid backslashes
              4. Trailing garbage character trimming (e.g. ]], ], >, })
              5. Outermost { ... } extraction
            """
            if not s or not isinstance(s, str):
                return None

            def _fix_escapes(src: str) -> str:
                return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', src)

            def _extract_candidates(raw: str) -> list[str]:
                cands = [raw, _fix_escapes(raw)]
                st = raw.strip()
                if st.startswith("{{") and st.endswith("}}"):
                    cands.extend([st[1:-1], _fix_escapes(st[1:-1])])
                elif st.startswith("{{"):
                    cands.extend([st[1:], _fix_escapes(st[1:])])
                elif st.endswith("}}"):
                    cands.extend([st[:-1], _fix_escapes(st[:-1])])

                # Trim extraneous trailing characters (e.g. ]] or >)
                trimmed = raw.strip()
                for _ in range(6):
                    if trimmed and trimmed[-1] in (']', '}', ')', '>', '`'):
                        trimmed = trimmed[:-1].strip()
                        cands.extend([trimmed, _fix_escapes(trimmed)])
                    else:
                        break

                # Match outermost { ... }
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if m:
                    matched = m.group(0)
                    cands.extend([matched, _fix_escapes(matched)])
                    if matched.startswith("{{") and matched.endswith("}}"):
                        cands.extend([matched[1:-1], _fix_escapes(matched[1:-1])])
                return cands

            for cand in _extract_candidates(s.strip()):
                try:
                    val = json.loads(cand)
                    if isinstance(val, dict):
                        return val
                except (json.JSONDecodeError, ValueError):
                    continue
            return None

        # Split on <tool_call> as delimiter — do NOT rely on closing tags.
        # This prevents a missing </tool_call> from fusing multiple calls into one block.
        raw_blocks = re.split(r"<tool_call>", text)
        # raw_blocks[0] is text before first <tool_call>, skip it
        processed_matches = []
        for raw_block in raw_blocks[1:]:
            # Strip optional closing tag and surrounding whitespace
            block = re.sub(r"\s*</tool_call>\s*$", "", raw_block).strip()
            processed_matches.append(block)

        for block in processed_matches:
            if not block:
                continue

            parsed: dict | None = None
            candidates = [
                block,
                block[1:] if block.startswith("{{") else block,
                block[1:-1] if block.startswith("{{") and block.endswith("}}") else block,
                block[:-1] if block.endswith("}}") else block,
            ]

            for cand in candidates:
                parsed = _lenient_loads(cand)
                if parsed:
                    break

            if parsed and "name" in parsed:
                args = parsed.get("arguments", {})
                if isinstance(args, str):
                    # args may be a stringified JSON object — try progressively:
                    # 1. parse directly, 2. strip {{}} wrappers, 3. deep-decode via json.loads twice
                    decoded: dict | None = None
                    for ac in [
                        args,
                        args[1:] if args.startswith("{{") else args,
                        args[:-1] if args.endswith("}}") else args,
                        args[1:-1] if args.startswith("{{") and args.endswith("}}") else args,
                    ]:
                        decoded = _lenient_loads(ac)
                        if decoded:
                            break
                    # Last resort: try double-decoding (string that is itself JSON-encoded JSON)
                    if not decoded:
                        try:
                            once = json.loads(args)
                            if isinstance(once, str):
                                decoded = _lenient_loads(once)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    args = decoded if decoded else {}

                if not isinstance(args, dict):
                    args = {}

                args.pop("text", None)  # de-bloat: strip document copy
                tool_calls.append({
                    "name": parsed["name"],
                    "args": args,
                })

        # Strip tool_call XML blocks from remaining conversational text
        if tool_calls:
            text = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text, flags=re.DOTALL).strip()


    return {"text": text, "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# LLMCallable builders
# ---------------------------------------------------------------------------

def build_extraction_model(
    model_path: Path,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,   # -1 = all layers on GPU (Metal)
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> Any:
    """
    Build an LLMCallable for the ExtractionAgent.

    Uses llama-cpp-python native function calling so the model can emit
    structured tool invocations (search_text, normalize_date,
    normalize_currency) without prompt hacking.

    Args:
        model_path:    Path to the Qwen2.5-7B-Instruct Q4_K_M GGUF file.
        n_ctx:         Context window size (expanded to 8192 for multi-turn headroom).
        n_gpu_layers:  Layers to offload to Metal GPU (-1 = all).
        max_tokens:    Max tokens per completion.
        temperature:   Sampling temperature (low = more deterministic).

    Returns:
        LLMCallable — takes list[dict] transcript, returns {"text", "tool_calls"}.

    Raises:
        ImportError: If llama-cpp-python is not installed.
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is not installed.\n"
            "Install with Metal support:\n"
            '  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python'
        ) from exc

    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    openai_tools = _tool_specs_to_openai(TOOL_SPECS)

    def _call(transcript: list[dict]) -> dict:
        lc_transcript = _to_llama_transcript(transcript)
        response = llm.create_chat_completion(
            messages=lc_transcript,
            tools=openai_tools,
            tool_choice="auto",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _from_llama_response(response)

    return _call


def build_verification_model(
    model_path: Path,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    max_tokens: int = 512,
    temperature: float = 0.0,   # deterministic — classifier, not generator
) -> Any:
    """
    Build an LLMCallable for the VerificationAgent.

    No tool calling needed — the model outputs a single JSON decision.
    Lower temperature (0.0) enforces consistent classification behaviour.

    Args:
        model_path:   Path to the Qwen2.5-7B-Instruct Q4_K_M GGUF file.
        n_ctx:        Context window (smaller than extraction — simpler prompt).
        n_gpu_layers: Layers to offload to Metal GPU (-1 = all).
        max_tokens:   Max tokens per completion.
        temperature:  Sampling temperature.

    Returns:
        LLMCallable — takes list[dict] transcript, returns {"text", "tool_calls": []}.

    Raises:
        ImportError: If llama-cpp-python is not installed.
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is not installed.\n"
            "Install with Metal support:\n"
            '  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python'
        ) from exc

    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=VERBOSE_MODE,
    )

    def _call(transcript: list[dict]) -> dict:
        # Verification prompt is simple system+user — no tool history to convert
        messages = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in transcript
            if m["role"] in ("system", "user")
        ]
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response["choices"][0]["message"].get("content") or ""
        return {"text": content, "tool_calls": []}

    return _call


# ---------------------------------------------------------------------------
# EmbeddingCallable builder
# ---------------------------------------------------------------------------

def build_embedding_fn(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """
    Build an EmbeddingCallable backed by sentence-transformers.

    The model (80MB) is downloaded automatically to ~/.cache/huggingface
    on first call. Subsequent calls load from cache.

    VECTOR_DIM of all-MiniLM-L6-v2 is 384 — matches rag/vector_store.VECTOR_DIM.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        EmbeddingCallable — takes str, returns list[float] of length 384.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed.\n"
            "  pip install sentence-transformers"
        ) from exc

    _model = SentenceTransformer(model_name)

    def _embed(text: str) -> list[float]:
        return _model.encode(text, convert_to_numpy=True).tolist()

    return _embed


# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------

def download_qwen_gguf(
    dest_dir: Path = Path("models"),
    filename: str = "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    repo_id: str = "bartowski/Qwen2.5-7B-Instruct-GGUF",
) -> Path:
    """
    Download the Qwen2.5-7B-Instruct Q4_K_M GGUF file from HuggingFace Hub.

    Skips download if the file already exists at dest_dir/filename.

    Args:
        dest_dir: Local directory to save the model (default: models/).
        filename: GGUF filename on HuggingFace.
        repo_id:  HuggingFace repository ID.

    Returns:
        Path to the downloaded (or existing) GGUF file.

    Raises:
        ImportError: If huggingface_hub is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is not installed.\n"
            "  pip install huggingface_hub"
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = dest_dir / filename
    if local_path.exists():
        print(f"Model already present: {local_path}")
        return local_path

    print(f"Downloading {filename} from {repo_id} …")
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
    )
    return Path(cached)
