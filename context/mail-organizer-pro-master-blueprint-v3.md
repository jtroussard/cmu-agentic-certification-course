# Mail-Organizer-Pro: Master Technical Blueprint & AI Coding Assistant System Prompt

## 1. Architectural Overview

### 1.1 System Goal & Operating Environment
**Mail-Organizer-Pro** is a local, privacy-first, standalone document processing pipeline built in Python [1, 42]. Designed for individuals, home offices, and privacy-conscious professionals managing physical mail, utility bills, tax forms, and legal correspondence, the system automates physical and digital document ingestion. Inbound documents are ingested from local directories, deduplicated using SHA-256 byte hashes, parsed via text/OCR engines, validated against strict schemas, committed to a local SQLite database, and archived into deterministic directory structures [1, 3, 42, 45].

```
                     +---------------------------------------+
                     |        Incoming PDF / Scanned Mail    |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |    Ingestion & Invariant Guardrails   |
                     |  - SHA-256 Deduplication Check        |
                     |  - Text Slicing (First/Last 1k chars) |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |         Extraction Agent              |
                     |  - Progressive OCR / Text Parsing     |
                     |  - Custom ReAct Reasoning Loop        |
                     |  - Generates Intermediate JSON        |
                     |  - Field Confidence Scores            |
                     +---------------------------------------+
                                         |
                            (Pydantic Payload Handoff)
                                         v
                     +---------------------------------------+
                     |        Verification Agent             |
                     |  - Pydantic Schema Validation         |
                     |  - Confidence Threshold Check (>=0.85)|
                     |  - Local SQLite Entity Lookup         |
                     |  - RAG Fetch (Past Human Corrections) |
                     |  - Anchor Key Validation (Acct #)     |
                     |  - Immutable Path Mapping Lookup      |
                     +---------------------------------------+
                                    /         \
                       (Passes All) /           \ (Validation/Confidence Fail)
                                   v             v
            +---------------------------+   +----------------------------+
            | Deterministic Filesystem  |   |    Manual Review Queue     |
            | Archive & SQLite Commit   |   | (Human Intervention UI)    |
            +---------------------------+   +----------------------------+
                                                          |
                                            (User Corrects & Submits)
                                                          v
                                            +----------------------------+
                                            | Local Vector Store & DB    |
                                            |  - Structured Correction   |
                                            |    Payload Indexed         |
                                            +----------------------------+
```

### 1.2 Two-Agent Architecture
Responsibilities are partitioned cleanly between two explicit Python-based agents, each with a distinct role and constrained LLM usage:

1. **Extraction Agent (The Parser)**:
   - **Role**: A stateless, lightweight parser focused purely on understanding noisy, unstructured inputs [33, 34, 41].
   - **Operation**: Reads progressive OCR streams sliced strictly to the first 1,000 and last 1,000 characters (isolating critical identity markers like account numbers and dates while capping token usage) [34, 45]. Executes a custom ReAct reasoning loop to extract structured attributes, map them into an intermediate Pydantic JSON schema, and assign self-calibrated confidence scores ($0.0$ to $1.0$) to each extracted field [34, 46].
   - **Security Boundary**: Strictly prohibited from executing file mutations [32, 33].

2. **Verification Agent (The Classifier)**:
   - **Role**: A constrained LLM-as-classifier that resolves document routing using grounded enumeration, then acts as a policy enforcer and circuit breaker [34, 49].
   - **Operation**: Receives the Pydantic payload from the Extraction Agent. Validates schema compliance and confidence threshold ($\ge 0.85$). Queries SQLite for past human corrections via RAG. Then invokes an LLM with: (a) a hardcoded **L1 directory list** the LLM may never create or rename, (b) a live **directory tree dump** from SQLite for grounding, and (c) strict **L2+ extrapolation templates** constraining how new entity paths may be structured. The LLM outputs a `target_directory`, `new_filename`, and `requires_new_directory` flag — it selects or extrapolates, never free-generates.
   - **Human Approval Gate**: When `requires_new_directory = True`, the system surfaces a prompt: *"Agent wants to create `{proposed_path}` — Approve?"* If approved, the directory is created and committed to SQLite for future grounding. If rejected, a directory-selector modal allows the human to manually choose a destination (no convention checks applied).
   - **Security Boundary**: The LLM output is Python-validated against the L1 list and template patterns before any filesystem operation executes.

### 1.3 Custom ReAct Execution Loop vs. Dismissal of Tree-of-Thought (ToT)
- **Custom ReAct Loop Built from Scratch**: Rather than pulling in external orchestration frameworks or plugin abstractions, the system builds an explicit, custom `Reason -> Act -> Observe -> Reason` execution loop in Python [6, 7, 25]. The agentic loop manages state transitions, tool invocations, and text analysis directly, demonstrating a deep, hands-on implementation of agentic pattern matching [25, 26].
- **Explicit Dismissal of Tree-of-Thought (ToT)**: ToT branching and search pruning are officially excluded [20]. Mail organization is a linear pattern-matching and validation task, not a combinatorial state-space search [21, 22]. Applying ToT incurs exponential token costs, multi-pass latency bottlenecks, non-deterministic branching, and codebase bloat on local edge hardware [20, 23, 29]. Constrained ReAct combined with local RAG delivers sub-second, highly predictable execution with a lightweight footprint [26, 27, 54].

### 1.4 Local RAG Integration & Anchor Key Validation Guardrail
- **Human Correction Memory**: A lightweight local vector store operates alongside the relational SQLite database [14]. Native digital PDFs parse with high confidence and are excluded from vector indexing [17]. Only hard cases resolved by human manual review are indexed [14, 17].
- **Structured Correction Payload**: When a user corrects a document in the review UI, the system generates a payload pairing finalized metadata JSON with the 1,000-character OCR boundary slice, embedding it locally into the vector store [14, 50].
- **Retrieval & Anchor Key Validation**: When field confidence is low, the system invokes `retrieve_past_corrections(ocr_text: str, k: int = 1)` to fetch the single closest match [14, 18]. Before injecting retrieved context into the LLM, Python enforces a strict **Anchor Key Validation Rule**: it extracts unique alphanumeric anchors (e.g., account numbers or tax IDs) from both the current OCR and the retrieved record [19, 46]. If account numbers conflict, the retrieved context is rejected to prevent **retrieval drift** (e.g., misfiling a gas bill into an electric folder due to template layout similarity) [18, 19, 44].

---

## 2. Data Contracts

All data handoffs between agents utilize strictly typed Pydantic v2 schemas to ensure runtime verification, boundary safety, and static type enforcement [34, 38, 46].

```python
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class VerificationDecision(str, Enum):
    ARCHIVED = "ARCHIVED"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
    DEDUPLICATED = "DEDUPLICATED"


class AnchorKeyStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class FieldMetadata(BaseModel):
    value: Optional[str] = Field(
        default=None, 
        description="Extracted value normalized to string."
    )
    confidence: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Self-calibrated confidence score between 0.0 and 1.0."
    )
    extracted_raw: Optional[str] = Field(
        default=None, 
        description="Raw unparsed text snippet from OCR stream."
    )


class ExtractionAgentPayload(BaseModel):
    document_id: str = Field(description="Unique SHA-256 byte hash of input document.")
    file_path: str = Field(description="Source file path on local disk.")
    extracted_text_slice: str = Field(
        description="Composite slice of first 1,000 and last 1,000 OCR characters."
    )
    vendor: FieldMetadata = Field(description="Identified vendor or entity name.")
    account_number: FieldMetadata = Field(description="Unique account or tax ID anchor.")
    document_date: FieldMetadata = Field(description="Document date formatted as YYYY-MM-DD.")
    total_amount: FieldMetadata = Field(description="Total dollar amount or balance due.")
    document_type: FieldMetadata = Field(
        description="Document category e.g., utility_bill, tax_form, legal, physical_mail."
    )
    batch_number: Optional[str] = Field(
        default=None, 
        description="Optional physical scan batch identifier."
    )
    overall_confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Minimum score across mandatory extraction fields."
    )

    @field_validator("overall_confidence", mode="before")
    def compute_overall_confidence(cls, v, info):
        # Fallback helper to enforce minimum confidence score across critical fields
        return v


class RAGContextPayload(BaseModel):
    retrieved_record_id: str
    retrieved_similarity_score: float
    retrieved_vendor: str
    retrieved_account_number: str
    suggested_destination_path: str
    anchor_key_match: bool


class VerificationInput(BaseModel):
    extraction_payload: ExtractionAgentPayload
    rag_context: Optional[RAGContextPayload] = None


class VerificationResult(BaseModel):
    document_id: str
    decision: VerificationDecision
    destination_path: Optional[str] = None
    reason: str
    anchor_key_status: AnchorKeyStatus
    confidence_check_passed: bool
    execution_timestamp: str
```

---

## 3. Coding Standards

All generated code for Mail-Organizer-Pro must strictly conform to these engineering standards:

1. **Single Responsibility Principle (SRP) & DRY**:
   - Each module, class, and function must have exactly one operational responsibility.
   - Isolate file I/O, byte hashing, LLM prompt generation, Pydantic validation, vector querying, and SQLite transactions into modular files.
   - Eliminate duplicated regex patterns, SQL connection setups, and error handling logic.

2. **Readability & Auditability**:
   - Code must be clear, clean, and fully annotated with explicit variable names.
   - Enforce explicit type annotations across all function and method signatures (`def process_document(path: Path) -> ExtractionAgentPayload:`).
   - Avoid hidden global states, opaque lambda expressions, or undocumented magic numbers.

3. **Strict Typing & Static Verification**:
   - Code must pass `mypy --strict` without warnings.
   - Use Pydantic v2 models for data parsing and boundary enforcement across all inter-agent channels.
   - Catch specific exceptions explicitly (`try...except FileNotFoundError`).

4. **Isolated Unit Testing**:
   - Every core module must include isolated unit tests using `pytest`.
   - Mock all LLM calls, vector database reads, and filesystem moves to ensure test suite execution remains deterministic and sub-second.
   - Unit tests must cover all core pipeline edge cases:
     - Duplicate file detection via SHA-256 byte hashing [45].
     - OCR text buffer slicing boundary enforcement (first 1,000 + last 1,000 characters) [17, 45].
     - Pydantic validation failures on missing required fields [34, 46].
     - Anchor Key matching pass/fail logic (matching vs mismatched account numbers) [19, 46].
     - Verification Agent circuit breaker routing when field confidence falls below $0.85$ [49].

---

## 4. Assistant Execution Rules

```
You are operating under a tight two session deadline. Be super pragmatic and laser focused on the immediate deliverable. Do not go down rabbit holes or suggest over engineered theoretical solutions. No walls of text. Provide code and concise explanations in a strict step by step manner. You must wait for human confirmation and test results before moving to the next build phase.
```

---

## 5. Step-By-Step Build Plan

The project execution is broken down into discrete, testable chunks optimized for a fast, high-velocity development sprint:

```
+-------------------------------------------------------------------------------+
| PHASE 1: Data Contracts, Core Utilities & Database Schema                     |
| - Define Pydantic V2 Models (ExtractionPayload, VerificationResult)           |
| - Create SQLite Manifest DB Schema & Connection Layer                         |
| - Build SHA-256 File Hashing & Deduplication Module                           |
| - Write Unit Tests for Schemas, Hashing & SQLite CRUD                         |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| PHASE 2: Ingestion & Text Boundary Slicing Pipeline                           |
| - Implement Local PDF Text Parser & Fallback OCR Engine                       |
| - Implement Text Slicer (First 1,000 + Last 1,000 characters)                 |
| - Write Unit Tests for Text Extraction & Slicing Invariants                   |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| PHASE 3: Extraction Agent & Custom ReAct Reasoning Loop                       |
| - Implement Extraction Agent with Custom Python ReAct Loop                    |
| - Implement Self-Calibrated Field Confidence Scoring Logic                    |
| - Write Unit Tests (Mock LLM Outputs to test JSON schema parsing)            |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| PHASE 4: Local RAG Memory & Anchor Key Guardrail                              |
| - Build Local Vector Index for Structured Correction Payloads                 |
| - Implement retrieve_past_corrections(ocr_text, k=1) Tool                     |
| - Implement Python Anchor Key Validation (Account # Match Guardrail)          |
| - Write Unit Tests for Anchor Key Match/Mismatch Logic                        |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| PHASE 5: Verification Agent & Deterministic Routing Engine                    |
| - Implement Verification Agent Rules Engine (Confidence Check >= 0.85)         |
| - Implement Immutable Entity-to-Path Lookup Directory                         |
| - Build Circuit Breaker (Route to Needs_Review Queue on Failure)              |
| - Write Unit Tests for Circuit Breaker & Filesystem Routing                   |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| PHASE 6: End-to-End Pipeline Integration & Human Review Loop                  |
| - Connect Ingestion -> Extraction -> Verification -> Archive / Manual Queue   |
| - Build Manual Review Callback (Generates Correction Payload & Updates Vector)|
| - Execute End-to-End Integration Test Suite                                   |
+-------------------------------------------------------------------------------+
```

### Sprint Deliverables

- **Phase 1: Environment, Contracts & Database**: Construct `models/schemas.py` for data handoffs, `utils/hashing.py` for SHA-256 deduplication, `db/database.py` for SQLite connection/schema management, and `tests/test_phase1.py` for schema/hashing tests.
- **Phase 2: Ingestion & Text Slicing**: Construct `ingestion/parser.py` for PDF/OCR ingestion, `ingestion/slicer.py` enforcing the 1,000-character header/footer boundary, and `tests/test_phase2.py` for text boundary slicing verification.
- **Phase 3: Extraction Agent & Custom ReAct Loop**: Construct `agents/extraction_agent.py` implementing the explicit Python ReAct loop and schema binding, and `tests/test_phase3.py` using mocked LLM payloads to verify field confidence scoring and JSON serialization.
- **Phase 4: Local RAG & Anchor Key Guardrail**: Construct `rag/vector_store.py` for local vector storage, `rag/anchor_key.py` for Python account number matching, and `tests/test_phase4.py` verifying context rejection on mismatched anchors.
- **Phase 5: Verification Agent & Routing Engine**: Construct `agents/verification_agent.py` implementing the LLM-as-classifier with grounded enumeration (L1 hardcoded list + live SQLite dir tree + L2+ templates), `config/l1_directories.py` defining the hardcoded L1 roots, `storage/archiver.py` executing atomic file moves, `review_queue/approval_prompt.py` implementing the human approval gate (approve / directory-selector fallback), and `tests/test_phase5.py` testing circuit breaker fallbacks and approval gate routing.
- **Phase 6: Integration & Human Review Pipeline**: Construct `pipeline.py` to orchestrate end-to-end execution, `review_queue/handler.py` to convert human corrections into vector embeddings, and `tests/test_integration.py` for full system validation.

---

## 6. Tech Stack

**Guiding Principles:** Local-first, edge-compute optimized, highly deterministic, and strictly bound by Pydantic data contracts. No heavy orchestration frameworks (LangChain, CrewAI) and no mandatory cloud models.

> **Note on Model Flexibility:** Model selection is not fixed. If a chosen model presents integration barriers or performance issues during development, swap to an alternative without ceremony. Priority order: (1) meet capstone requirements, (2) working code, (3) development speed. Model swaps do not require blueprint revision.

---

### 6.1 AI Models (Hugging Face / GGUF)

Quantized models (GGUF format, 4-bit or 8-bit precision) enable local inference on consumer hardware without prohibitive latency.

#### Extraction Agent (The Parser)
**Role:** Strong JSON-generation, tool-calling for the custom ReAct loop, handles messy OCR text slices.

| Priority | Model | Notes |
|---|---|---|
| ✅ Primary | [`Qwen/Qwen2.5-3B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) | Leads local benchmarks for structured JSON output & instruction following. ~3-5GB RAM. |
| Alt 1 | [`meta-llama/Llama-3.2-3B-Instruct-GGUF`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | Fast, portable, tuned for tool calling. |
| Alt 2 | [`microsoft/Phi-3.5-mini-instruct-GGUF`](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) (3.8B) | Strong reasoning for its size; good CPU-only fallback. |

#### Verification Agent (The Classifier)
- **Model: Same options as Extraction Agent** (Qwen2.5-3B primary, see table above).
- **Usage pattern:** LLM-as-classifier only. Receives L1 hardcoded dir list + live SQLite directory tree + L2+ extrapolation templates. Outputs `target_directory`, `new_filename`, `requires_new_directory`. Python validates output before any filesystem write.
- Swap policy: same as Extraction Agent — change without ceremony if barriers arise.

#### Embedding Model (Local RAG Memory)
| Priority | Model | Notes |
|---|---|---|
| ✅ Primary | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | ~130MB, CPU-only, excellent single-doc ($k=1$) retrieval. Embeds the 1,000-char OCR boundary slices. |

---

### 6.2 Python Orchestration & Data Contracts

| Component | Choice | Reason |
|---|---|---|
| ReAct Loop | Custom Python `while` loop | Demonstrates agentic pattern understanding; avoids framework bloat |
| Data Boundaries | `Pydantic v2` | Backbone of all inter-agent handoffs; enforces confidence thresholds at runtime |
| LLM Inference | `llama-cpp-python` | Runs GGUF in-process; native JSON Schema enforcement mathematically constrains model output |

---

### 6.3 State Management, RAG & Storage

| Component | Choice | Reason |
|---|---|---|
| Relational Storage | `sqlite3` (stdlib) | Single-file, zero-dependency; handles deduplication manifest, entity-to-path lookup, review queue state |
| Vector Database (RAG) | `sqlite-vec` | C-extension that adds vector search to SQLite; relational + RAG embeddings in one `.sqlite` file — no ChromaDB/FAISS |

---

### 6.4 Ingestion & Perception Tools

| Component | Choice | Reason |
|---|---|---|
| Digital PDF Extraction | `PyMuPDF` (`fitz`) | C-based, millisecond text extraction from pristine PDFs; no OCR overhead |
| Physical Scan OCR | `pytesseract` (Tesseract) | Free, fully offline, GPU-free fallback for image-only scans |
| SHA-256 Deduplication | `hashlib` (stdlib) | Zero-dependency byte hashing; deduplication guardrail runs before any LLM call |

---

### 6.5 `requirements.txt`

```text
# --- Core Data Contracts ---
pydantic>=2.5.0         # Boundary enforcement and payload schemas

# --- AI & Execution ---
llama-cpp-python        # In-process GGUF runner with JSON schema enforcement
sentence-transformers   # Runs BGE-small embeddings locally

# --- Database & State ---
sqlite-vec              # Vector search embedded directly into SQLite

# --- Ingestion Tools ---
PyMuPDF                 # Native digital PDF text extraction
pytesseract             # OCR fallback for physical scans
pillow                  # Image processing dependency for pytesseract

# --- Testing ---
pytest                  # Isolated unit tests with mocked LLM outputs
```

