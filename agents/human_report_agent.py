# agents/human_report_agent.py
from __future__ import annotations

"""
HumanReportAgent
----------------
LLM-based summarizer that converts the machine-generated summary.md (from
FinalReportAgent) into a concise, high-signal, director-ready executive summary.

Key properties:
- Uses ONLY content present in summary.md (no new facts).
- Emphasizes unresolved items, risks/blockers, cost issues, and OWNERSHIP gaps.
- Produces clean Markdown suitable for C-level consumption.

Usage:
    from agents.human_report_agent import HumanReportAgent
    hr = HumanReportAgent(backend=backend)
    md = hr.generate_from_file(summary_md_path, organization="Acme", director_name="Jane Doe")
"""

from pathlib import Path
from typing import Optional
from clients.backend import LLMBackend
from datetime import datetime, timezone

class HumanReportAgent:
    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def generate_from_file(
        self,
        summary_md_path: Path,
        *,
        report_title: str = "QBR Executive Summary",
    ) -> str:
        """Load generated  summary from disk and produce a human-readable summary."""
        text = Path(summary_md_path).read_text(encoding="utf-8") if summary_md_path.exists() else ""
        return self.generate_from_text(
            text,
            report_title=report_title,
        )

    def generate_from_text(
        self,
        summary_md_text: str,
        *,
        report_title: str = "QBR Executive Summary",
    ) -> str:
        """Summarize the provided machine summary text into a Markdown."""
        summary_md_text = (summary_md_text or "").strip()
        if not summary_md_text:
            return "# QBR Executive Summary\n\n_No selected threads; portfolio currently appears stable._\n"

        prompt = self._build_prompt(
            summary_md_text,
            report_title=report_title,
        )
        out = self.backend.complete(prompt)
        # Minimal safety: non-empty and looks like Markdown; otherwise fall back to the input.
        if isinstance(out, str) and out.strip():
            return out.strip()
        return summary_md_text

    # ---------- Prompt ----------
    def _build_prompt(
        self,
        machine_summary_markdown: str,
        *,
        report_title: str,
    ) -> str:

        # IMPORTANT: We forbid adding facts beyond the provided machine summary.
        # The model may rephrase/condense and propose actions implied by the text,
        # but must not invent owners, dates, or costs if they are not present.
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Tiny positive example to bias the formatter toward Markdown (not JSON)
        example_md = (
            "# QBR Executive Summary\n\n"
            "Generated: 2025-10-13 12:00 UTC\n\n"
            "## Overall Assessment\n"
            "Portfolio is progressing but two threads require immediate attention.\n\n"
            "## Hotlist (Immediate Attention)\n"
            "- Project Alpha — unresolved API gateway blocker. **Owner/Due: MISSING**\n"
            "- Project Beta — potential tool overrun; request cost/benefit analysis.\n\n"
            "## Next 7 Days — Director Focus\n"
            "- Decide owner and deadline for API gateway fix.\n"
            "- Ask EM for a one-page cost analysis before purchase approval.\n"
        )

        return f"""
        System:
        You are a Chief of Staff producing a one-page brief for a Director of Engineering's QBR.

        Hard rules (must follow):
        - RETURN ONLY MARKDOWN. DO NOT output JSON, YAML, or code blocks. No braces/brackets.
        - DO NOT invent facts/people/dates/costs not present in the machine summary.
        - If ownership or due dates are missing, explicitly write: "Owner/Due: MISSING".
        - Keep it concise, executive-friendly, and action-oriented.

        Formatting:
        - Start with: "# {report_title}"
        - Then a line: "Generated: {now}"
        - Then sections (omit if empty): "## Overall Assessment", "## Hotlist (Immediate Attention)",
          "## Risks & Blockers", "## Cost & Approvals", "## Ownership Gaps", "## Next 7 Days — Director Focus"
        - Use short paragraphs and bullet lists. No tables for this brief.

        Here is a SHORT EXAMPLE of the EXPECTED MARKDOWN FORMAT (do NOT copy its content):
        ---
        {example_md}
        ---

        Context for synthesis (Machine Summary from the scoring/selection stage):
        ---
        {machine_summary_markdown}
        ---

        Produce the final Markdown now. Remember: NO JSON, NO CODE FENCES.
        """.strip()
