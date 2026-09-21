"""
agents/verification_agent.py

Verification Agent: LLM-as-classifier that routes an ExtractionAgentPayload
to the correct destination directory using a constrained decision space.

Design constraints:
  - The LLM is shown only the KNOWN directory list from SQLite — it cannot
    invent a free-form path.  It either picks an existing path or sets
    requires_new_directory = True to surface a human approval gate.
  - If overall_confidence < CONFIDENCE_THRESHOLD the document bypasses the
    LLM entirely and is sent straight to NEEDS_HUMAN_REVIEW.
  - If a RAG hit with anchor_key_match=True is present, its
    suggested_destination_path is used as a strong prior (no LLM call needed).
  - Filename is generated deterministically from extracted fields:
    YYYY-MM-DD_ENTITY_DOCUMENTTYPE.ext  (no LLM involvement).

LLMCallable is injected — same pattern as ExtractionAgent.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

from models.schemas import (
    AnchorKeyStatus,
    ExtractionAgentPayload,
    RAGContextPayload,
    VerificationDecision,
    VerificationInput,
    VerificationResult,
)
from db.database import load_directory_taxonomy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD: float = 0.85
ROUTING_CONFIDENCE_THRESHOLD: float = 0.85

APPROVED_L1_ROOTS: set[str] = {
    "01_Finance",
    "02_Home",
    "03_Vehicles",
    "04_Pets",
    "05_Legal",
    "06_Employment",
    "07_Insurance",
    "08_Travel",
    "09_Personal",
    "10_Rental_Property",
    "11_Taxes",
    "90_Archive",
    "99_Unsorted",
}

LLMCallable = Callable[[list[dict]], dict]


# ---------------------------------------------------------------------------
# Verification Agent
# ---------------------------------------------------------------------------

class VerificationAgent:
    """
    Routes a document to a destination directory using LLM-as-classifier.

    The LLM receives the document metadata and a constrained list of known
    directories.  It returns a JSON decision — either picking an existing path
    or flagging that a new directory is required (triggering the human gate).
    """

    def __init__(self, model: LLMCallable) -> None:
        """
        Args:
            model: Callable accepting a transcript and returning a response dict
                   with "text" key containing JSON.
        """
        self._model = model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        verification_input: VerificationInput,
        known_dirs: list[str],
    ) -> VerificationResult:
        """
        Execute the verification and routing decision.

        Decision flow:
          1. Confidence gate — below threshold → NEEDS_HUMAN_REVIEW immediately.
          2. RAG gate — anchor-verified RAG hit → use stored path directly.
          3. LLM classifier — ask the model to pick from known_dirs.
          4. New directory flag — requires_new_directory triggers human gate.
          5. Filename generation — deterministic, no LLM involvement.

        Args:
            verification_input: Pydantic payload from the Extraction Agent
                                 plus optional RAG context.
            known_dirs:         List of known directory paths from SQLite
                                (passed in from the pipeline, not fetched here
                                to keep this module I/O-free and testable).

        Returns:
            VerificationResult with decision, destination path, and filename.
        """
        payload = verification_input.extraction_payload
        rag = verification_input.rag_context

        confidence_passed = payload.overall_confidence >= CONFIDENCE_THRESHOLD
        anchor_status = self._resolve_anchor_status(rag)
        timestamp = datetime.now(timezone.utc).isoformat()

        # --- Gate 1: Anchor-verified RAG hit → use stored path directly ---
        if rag and rag.anchor_key_match:
            return VerificationResult(
                document_id=payload.document_id,
                decision=VerificationDecision.ARCHIVED,
                destination_path=rag.suggested_destination_path,
                new_filename=self._generate_filename(payload),
                requires_new_directory=False,
                reason=(
                    f"RAG hit with anchor key match (similarity "
                    f"{rag.retrieved_similarity_score:.2f}). "
                    f"Using stored path: {rag.suggested_destination_path}"
                ),
                anchor_key_status=anchor_status,
                confidence_check_passed=True,
                execution_timestamp=timestamp,
            )

        # --- Gate 2: Low confidence → review queue ---
        if not confidence_passed:
            return VerificationResult(
                document_id=payload.document_id,
                decision=VerificationDecision.NEEDS_HUMAN_REVIEW,
                destination_path=None,
                new_filename=None,
                requires_new_directory=False,
                reason=(
                    f"overall_confidence {payload.overall_confidence:.2f} is below "
                    f"threshold {CONFIDENCE_THRESHOLD:.2f}. Manual review required."
                ),
                anchor_key_status=anchor_status,
                confidence_check_passed=False,
                execution_timestamp=timestamp,
            )

        # --- Gate 3: LLM classifier (with optional RAG memory injection) ---
        suggested_path, requires_new_dir, llm_reason, entity_part, desc_part, routing_confidence = self._classify_path(
            payload, known_dirs, rag
        )

        routing_conf_passed = (routing_confidence >= ROUTING_CONFIDENCE_THRESHOLD)
        if not routing_conf_passed and suggested_path:
            llm_reason = f"Routing confidence ({routing_confidence:.2f}) is below threshold ({ROUTING_CONFIDENCE_THRESHOLD:.2f}). {llm_reason}"

        decision = (
            VerificationDecision.NEEDS_HUMAN_REVIEW
            if (requires_new_dir or not suggested_path or not routing_conf_passed)
            else VerificationDecision.ARCHIVED
        )

        new_filename = self._generate_filename(payload, entity_part, desc_part)

        return VerificationResult(
            document_id=payload.document_id,
            decision=decision,
            destination_path=suggested_path,
            new_filename=new_filename,
            requires_new_directory=requires_new_dir,
            reason=llm_reason,
            anchor_key_status=anchor_status,
            confidence_check_passed=confidence_passed and routing_conf_passed,
            execution_timestamp=timestamp,
            routing_confidence=routing_confidence,
        )

    # ------------------------------------------------------------------
    # LLM classification
    # ------------------------------------------------------------------

    def _classify_path(
        self,
        payload: ExtractionAgentPayload,
        known_dirs: list[str],
        rag: Optional["RAGContextPayload"] = None,
    ) -> tuple[Optional[str], bool, str, Optional[str], Optional[str], float]:
        """
        Call the LLM as a constrained classifier to pick a destination path.

        When a RAG memory hit is present, its past human correction is injected
        into the system prompt as a strong prior before the LLM reasons.

        The LLM may ONLY select from known_dirs or propose a new directory
        by setting requires_new_directory=True.

        Returns:
            Tuple of (suggested_path, requires_new_directory, reason, entity_part, desc_part, routing_confidence).
        """
        prompt = self._build_classifier_prompt(payload, known_dirs, rag)
        response = self._model([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Classify this document and return your JSON decision."},
        ])

        raw = response.get("text", "")
        text_snippet = payload.extracted_text_slice if payload.extracted_text_slice else ""
        return self._parse_classifier_output(raw, known_dirs, text_snippet=text_snippet)

    def _build_classifier_prompt(
        self,
        payload: ExtractionAgentPayload,
        known_dirs: list[str],
        rag: Optional["RAGContextPayload"] = None,
    ) -> str:
        dirs_formatted = "\n".join(f"  - {d}" for d in known_dirs)
        doc_id_val = getattr(payload, "document_identifier", None).value if getattr(payload, "document_identifier", None) else None
        metadata = {
            "entity": payload.entity.value,
            "account_number": payload.account_number.value,
            "document_identifier": doc_id_val,
            "document_date": payload.document_date.value,
            "document_type": payload.document_type.value,
            "total_amount": payload.total_amount.value,
        }
        text_snippet = payload.extracted_text_slice[:1200] if payload.extracted_text_slice else ""

        # Build RAG memory block if a trusted hit was retrieved
        rag_memory_block = ""
        if rag is not None:
            rag_memory_block = f"""
CRITICAL MEMORY FROM PAST HUMAN CORRECTIONS:
A human previously reviewed a structurally similar document and made this routing decision:
  - Vendor/Entity: {rag.retrieved_entity}
  - Document Type: {rag.retrieved_account_number}
  - Routed to: {rag.suggested_destination_path}
  - Similarity score: {rag.retrieved_similarity_score:.2f}

Apply this past human decision as your STRONG PRIOR. Select the same destination path
unless the current document's content clearly contradicts it (different property address,
different L1 category, etc.).
"""

        taxonomy = load_directory_taxonomy()
        home_tax = taxonomy.get("02_Home", {})
        home_tooltip = home_tax.get("category_tooltip", "Repairs = fixing broken items; Maintenance = routine upkeep/pest control; Renovations = structural/aesthetic upgrades; Utilities = recurring services.")
        vehicles_tax = taxonomy.get("03_Vehicles", {})
        vehicles_tooltip = vehicles_tax.get("category_tooltip", "Repairs = fixing defects; Maintenance = oil changes/inspections; DMV = registration/title.")

        return f"""You are a rigid, deterministic file classification and taxonomy agent. Your sole purpose is to analyze document metadata and extracted text from ingested mail/documents, select or construct the exact destination directory, and provide structured parts for canonical filename generation.
{rag_memory_block}
TAXONOMY & ROUTING RULES:
1. Level 1 (L1) roots are strictly: 01_Finance, 02_Home, 03_Vehicles, 04_Pets, 05_Legal, 06_Employment, 07_Insurance, 08_Travel, 09_Personal, 10_Rental_Property, 11_Taxes, 90_Archive, 99_Unsorted.

2. 02_Home (Physical Properties):
   - Pattern: 02_Home/{{Property_Address_City_ST}}/{{Category}}/
   - Categories: Purchase, Mortgage, Taxes, Insurance, Repairs, Renovations, Maintenance, Utilities, Contractors, Warranties.
   - Subcategories: Utilities/Water, Utilities/Electric, Utilities/Gas, Utilities/Internet.
   - Taxonomy Guidance: {home_tooltip}
   - CRITICAL PROPERTY GROUNDING RULE: If a bill, utility, or notice mentions a specific physical property service address (e.g. "45 CRAIGHEAD ST"), it MUST route to 02_Home/{{Property_Address_City_ST}}/... NEVER 01_Finance.
     HOWEVER: To route into any specific property directory under 02_Home/{{Property_Address_City_ST}}/..., the document text MUST explicitly state that specific street address.
     If the document ONLY mentions a general city or state (e.g. "Pittsburgh, PA") without a street address, or if the property address is missing or ambiguous, you MUST NOT assume or guess a specific property address with high confidence!
     In that case, set "routing_confidence": 0.50 (strictly below 0.85) and state in "reason" that the property street address is missing or ambiguous.

   - TRADE SKILLS VS. WORK PERFORMED: Trades (e.g. Plumbing, Electrical, HVAC, Roofing) indicate the contractor trade, not the category of work. Determine the category from the work performed:
     * "Repairs": Corrective fixes, leak repairs, replacing broken/failed/defective components, and restoring damaged systems.
     * "Maintenance": Preventive upkeep, routine cleaning, seasonal servicing, and scheduled inspections to keep functioning systems operational.
     * "Renovations": Structural or aesthetic upgrades, additions, and remodels.
     * "Contractors": General agreements, master contracts, or broad estimates not tied to a specific repair or maintenance task.

   - WORK TYPE REASONING STRATEGY & CONFIDENCE SCORING:
     1. Holistic Review: Read every word in the description and line items, then consider the description as a whole to evaluate the true scope of work. Do not rely on isolated words alone.
     2. Contextual Indicators (Hints to guide understanding, not rigid rules):
        - Repair indicators: Words like "replace", "repaired", "installed", "fixed", "rebuilt", "reconnected", or "leak" typically indicate corrective replacement or repairs.
        - Maintenance indicators: Verbs and terms like "cleaned", "inspected", "snaked", "serviced", "tuned up", "cleared", or "mowed" typically indicate routine upkeep.
     3. Honest Confidence Calibration ("routing_confidence" between 0.0 and 1.0):
        - 0.90 – 1.00: The document description, work performed, AND property address unambiguously match without doubt.
        - 0.60 – 0.84: The description is terse, sparse, or borderline between categories (e.g. unclear whether a job was routine servicing or component replacement).
        - < 0.60 (e.g. 0.50): The property street address is missing or cannot be verified from the document text.
        - DO NOT artificially inflate confidence to appear certain. When the description or property address is ambiguous or missing, score routing_confidence honestly (< 0.85) so a human can confirm.

3. 03_Vehicles:
   - Pattern: 03_Vehicles/{{YYYY_Make_Model}}/{{Category}}/
   - Taxonomy Guidance: {vehicles_tooltip}

4. 01_Finance:
   - General financial accounts, banking, investments. NOT property-specific utility bills.

5. DIRECTORY SELECTION & NEW DIRECTORY GATE:
   - Check the KNOWN DIRECTORIES list below first.
   - If an existing directory matches the property AND the EXACT category determined above (e.g. "02_Home/45_Craighead_St_Pittsburgh_PA/Repairs" for a repair), select it and set "requires_new_directory": false.
   - CRITICAL CONSTRAINT: NEVER force or collapse a "Repairs" invoice into an existing "Maintenance" directory (or vice versa) simply because "Maintenance" is already in KNOWN DIRECTORIES.
   - If the correct category directory does NOT yet exist in KNOWN DIRECTORIES (e.g. "02_Home/45_Craighead_St_Pittsburgh_PA/Repairs" is not in KNOWN DIRECTORIES), you MUST propose it and set "requires_new_directory": true.

DOCUMENT METADATA:
{json.dumps(metadata, indent=2)}

DOCUMENT TEXT EXCERPT:
\"\"\"
{text_snippet}
\"\"\"

KNOWN DIRECTORIES:
{dirs_formatted}

FILENAME CONVENTION (zero spaces, underscores only):
- entity_part: Issuing vendor/organization (e.g. "PWSA", "Spectrum_Pest_Control", "Duquesne_Light"). No spaces.
- description_part: Concise document summary (e.g. "Water_Bill", "Service_Report", "Invoice"). No spaces.

OUTPUT: Return ONLY this JSON, no explanation:
{{
  "suggested_path": "02_Home/45_Craighead_St_Pittsburgh_PA/Maintenance",
  "entity_part": "Spectrum_Pest_Control",
  "description_part": "Service_Report",
  "requires_new_directory": false,
  "routing_confidence": 0.95,
  "reason": "one sentence explanation"
}}"""

    def _parse_classifier_output(
        self,
        raw: str,
        known_dirs: list[str],
        text_snippet: Optional[str] = None,
    ) -> tuple[Optional[str], bool, str, Optional[str], Optional[str], float]:
        """
        Parse the LLM's JSON response and run Critic loop verifications.

        Validates that suggested_path starts with an approved L1 root,
        and matches known_dirs when requires_new_directory=False.
        Falls back to NEEDS_HUMAN_REVIEW on parse failure or invalid path selection.
        """
        try:
            json_str = raw.strip()
            if "```" in json_str:
                parts = json_str.split("```")
                json_str = parts[1].lstrip("json").strip() if len(parts) > 1 else json_str
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return None, False, "LLM returned malformed JSON — routed to human review.", None, None, 0.0

        suggested = data.get("suggested_path") or data.get("target_directory")
        requires_new = bool(data.get("requires_new_directory", False))
        reason = str(data.get("reason", "No reason provided."))
        entity_part = data.get("entity_part")
        desc_part = data.get("description_part")

        routing_confidence_raw = data.get("routing_confidence")
        try:
            routing_confidence = float(routing_confidence_raw) if routing_confidence_raw is not None else 1.0
            routing_confidence = max(0.0, min(1.0, routing_confidence))
        except (ValueError, TypeError):
            routing_confidence = 1.0

        if isinstance(suggested, str):
            suggested = suggested.strip()

        if not suggested:
            return None, False, "LLM returned empty suggested_path — routed to human review.", entity_part, desc_part, routing_confidence

        # Critic Check 1: Must start with an approved L1 root
        suggested_clean = suggested.strip("/")
        l1_prefix = suggested_clean.split("/")[0]
        if l1_prefix not in APPROVED_L1_ROOTS:
            return (
                suggested,
                True,
                f"Path '{suggested}' does not start with an approved L1 root. Human review required.",
                entity_part,
                desc_part,
                routing_confidence,
            )

        # Critic Check 2: State Grounding against known_dirs
        norm_map = {d.strip("/"): d for d in known_dirs}
        if suggested_clean in norm_map:
            suggested = norm_map[suggested_clean]
            requires_new = False
        else:
            if not requires_new:
                reason = f"LLM suggested '{suggested}' which is not in known_dirs — human approval required."
            requires_new = True

        # Critic Check 3: Property Address Grounding for 02_Home
        if l1_prefix == "02_Home" and len(suggested_clean.split("/")) >= 2 and text_snippet:
            prop_segment = suggested_clean.split("/")[1]
            generic_stop = {
                "st", "street", "ave", "avenue", "rd", "road", "dr", "drive", "ln", "lane",
                "ct", "court", "blvd", "boulevard", "pittsburgh", "pa", "apt", "unit", "suite",
                "north", "south", "east", "west", "n", "s", "e", "w"
            }
            # Distinctive street name tokens (alphabetic tokens not in generic stop words)
            street_name_tokens = [
                tok.lower() for tok in prop_segment.split("_")
                if tok.isalpha() and tok.lower() not in generic_stop and len(tok) > 2
            ]
            if street_name_tokens:
                text_lower = text_snippet.lower()
                matches = [tok for tok in street_name_tokens if tok in text_lower]
                if not matches:
                    suggested = None
                    routing_confidence = 0.0
                    reason = f"Property street address is missing or not grounded in document text (cannot assume '{prop_segment}'). Destination property unknown; manual review required."

        return suggested, requires_new, reason, entity_part, desc_part, routing_confidence

    # ------------------------------------------------------------------
    # Filename generation — deterministic, no LLM
    # ------------------------------------------------------------------

    def _generate_filename(
        self,
        payload: ExtractionAgentPayload,
        entity_part: Optional[str] = None,
        desc_part: Optional[str] = None,
        anchor_key: Optional[str] = None,
        destination_path: Optional[str] = None,
        conn: Optional[Any] = None,
    ) -> str:
        """
        Generate a canonical filename from extracted fields following
        YYYY-MM-DD_ENTITY_DESCRIPTION_ANCHORKEY.ext convention.

        All spaces, hyphens, and ampersands are converted or sanitized.
        Zero spaces permitted.
        """
        if isinstance(payload, dict):
            date_raw = (payload.get("document_date") or "").strip()
            raw_entity = entity_part or payload.get("entity") or "Unknown_Entity"
            raw_desc = desc_part or payload.get("description") or payload.get("document_type") or "Document"
            key = anchor_key or payload.get("account_number") or payload.get("document_identifier")
            doc_id = payload.get("document_id") or "00000000"
            file_path = payload.get("file_path") or "document.pdf"
        else:
            date_raw = (payload.document_date.value or "").strip()
            raw_entity = entity_part or payload.entity.value or "Unknown_Entity"
            raw_desc = desc_part or payload.document_type.value or "Document"
            doc_id_val = payload.document_identifier.value if getattr(payload, "document_identifier", None) else None
            key = anchor_key or (payload.account_number.value if payload.account_number else None) or doc_id_val
            doc_id = payload.document_id
            file_path = payload.file_path

        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_raw):
            date_raw = "0000-00-00"

        clean_entity = self._sanitize_part(raw_entity)
        clean_desc = self._sanitize_part(raw_desc)

        def _is_real_key(k: Any) -> bool:
            return bool(k and str(k).strip() and str(k).strip().lower() not in ("no-account", "no-account-number", "none"))

        candidate_key = None
        doc_id_candidate = payload.get("document_identifier") if isinstance(payload, dict) else (payload.document_identifier.value if getattr(payload, "document_identifier", None) else None)
        for k in (key, doc_id_candidate):
            if _is_real_key(k):
                candidate_key = k
                break

        # Resolve anchor key: existing account/doc ID or generated synthetic key
        if candidate_key:
            clean_anchor = self._sanitize_part(str(candidate_key))
        else:
            prefix = self._resolve_domain_prefix(destination_path, raw_desc)
            if conn is not None:
                from db.database import generate_unique_anchor_key
                clean_anchor = generate_unique_anchor_key(conn, prefix, doc_id)
            else:
                hex_slice = doc_id[:8] if len(doc_id) >= 8 else "00000000"
                try:
                    seed = int(hex_slice, 16) % 10_000_000
                except ValueError:
                    seed = 1_000_000
                clean_anchor = f"{prefix}{seed:07d}"

        original_ext = Path(file_path).suffix.lower() or ".pdf"

        return f"{date_raw}_{clean_entity}_{clean_desc}_{clean_anchor}{original_ext}"

    @staticmethod
    def _sanitize_part(text: str) -> str:
        """Replace spaces, hyphens, ampersands, and quotes; strip non-alphanumeric except underscores."""
        text = text.strip()
        text = text.replace("&", "And")
        text = re.sub(r"['’\"`]", "", text)
        text = re.sub(r"[\s\-]+", "_", text)
        text = re.sub(r"[^a-zA-Z0-9_]", "", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("_") or "Unknown"

    @staticmethod
    def _resolve_domain_prefix(path_hint: Optional[str], doc_type: Optional[str]) -> str:
        """Map L1 directory or document type to standard single letter code."""
        text = f"{path_hint or ''} {doc_type or ''}".upper()
        if any(k in text for k in ("02_HOME", "HOME", "HOUSE", "PROPERTY")):
            return "H"
        if any(k in text for k in ("01_FINANCE", "BANK", "INVEST", "FINANCE")):
            return "F"
        if any(k in text for k in ("03_VEHICLES", "VEHICLE", "AUTO", "CAR")):
            return "V"
        if any(k in text for k in ("05_LEGAL", "LEGAL", "COURT", "CONTRACT")):
            return "L"
        if any(k in text for k in ("06_EMPLOYMENT", "EMPLOYMENT", "PAYSTUB", "JOB")):
            return "E"
        if any(k in text for k in ("11_TAXES", "TAX")):
            return "T"
        if any(k in text for k in ("09_PERSONAL", "PERSONAL", "LETTER")):
            return "P"
        return "M"

    def re_verify(
        self,
        payload: Union[ExtractionAgentPayload, dict],
        destination_path: str,
        conn: Optional[Any] = None,
    ) -> tuple[str, str]:
        """
        Abbreviated HITL post-edit re-review sub-routine.
        With human-confirmed fields, generates the final canonical
        filename following YYYY-MM-DD_ENTITY_DESCRIPTION_ANCHORKEY.ext.

        Returns:
            (canonical_destination_path, canonical_filename)
        """
        clean_path = destination_path.strip().strip("/")
        canonical_filename = self._generate_filename(
            payload=payload,
            destination_path=clean_path,
            conn=conn,
        )
        return clean_path, canonical_filename

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_anchor_status(rag: Optional[RAGContextPayload]) -> AnchorKeyStatus:
        """Map RAG context to the AnchorKeyStatus enum value."""
        if rag is None:
            return AnchorKeyStatus.NOT_APPLICABLE
        return AnchorKeyStatus.VERIFIED if rag.anchor_key_match else AnchorKeyStatus.FAILED
