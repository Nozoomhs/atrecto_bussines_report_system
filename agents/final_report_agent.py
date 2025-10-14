from __future__ import annotations

"""
FinalReportAgent
----------------
Aggregates the four per-thread agent outputs (unresolved, risks, cost, negligence),
computes attention scores, decides inclusion for the final report, and returns
normalized findings per thread. No LLM calls here—pure rules/weights.

Inputs per thread (from agent runners):
{
  "thread_id": "...",
  "agent": "unresolved|risk|cost|negligence",
  "result": { ... agent-specific JSON ... }
}

Optional metadata (if available):
- Threads.parquet with columns: [thread_id, first_seen_utc, last_seen_utc, participants]

Outputs:
- selected_threads: list[ReportThread]
- all_threads: list[ReportThread]
- helper to render a compact Markdown summary (optional for POC)
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timezone
from tools.string_tools import _safe_name

# ------------------------------- Models -------------------------------

@dataclass
class Evidence:
    message_id: str
    quote: str

@dataclass
class Finding:
    thread_id: str
    flag: str
    label: str
    confidence: float
    evidence: List[Evidence]
    attention_score: float = 0.0

@dataclass
class ThreadMeta:
    thread_id: str
    first_seen_utc: Optional[pd.Timestamp]
    last_seen_utc: Optional[pd.Timestamp]
    participants: Optional[List[str]]

@dataclass
class ReportThread:
    thread_id: str
    findings: List[Finding]
    meta: Optional[ThreadMeta]
    top_score: float


# -------------------------- FinalReportAgent --------------------------

class FinalReportParser:
    def __init__(
        self,
        threads_parquet: Optional[Path] = None,
        messages_parquet: Optional[Path] = None,
        people_parquet: Optional[Path] = None,
        recency_window_days: int = 30,
        open_window_days: int = 30,
        threshold: float = 0.55,
        top_k: Optional[int] = 20,
    ) -> None:
        self.threshold = threshold
        self.top_k = top_k
        self.recency_window_days = recency_window_days
        self.open_window_days = open_window_days
        self._meta: Dict[str, ThreadMeta] = {}

        # Try threads meta first
        if threads_parquet is not None and Path(threads_parquet).exists():
            self._load_thread_meta(Path(threads_parquet))



    # --------------------------- Public API ---------------------------
    def build_report(
        self, per_thread_agent_results: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Tuple[List[ReportThread], List[ReportThread]]:
        """
        Args:
            per_thread_agent_results: {thread_id: {agent_key: agent_output_json, ...}}
              agent_key in {"unresolved", "risk", "cost", "negligence"}
        Returns:
            (selected_threads, all_threads)
        """
        all_threads: List[ReportThread] = []
        for tid, agent_map in per_thread_agent_results.items():
            findings = self._normalize_and_score_thread(tid, agent_map)
            if not findings:
                continue
            top_score = max(f.attention_score for f in findings)
            meta = self._meta.get(str(tid))
            all_threads.append(ReportThread(thread_id=str(tid), findings=findings, meta=meta, top_score=top_score))

        # Decide inclusion
        selected = [t for t in all_threads if t.top_score >= self.threshold]
        selected.sort(key=lambda t: t.top_score, reverse=True)
        if self.top_k is not None:
            selected = selected[: self.top_k]
        return selected, all_threads

    @staticmethod
    def md_table(headers, rows):
        line1 = "| " + " | ".join(headers) + " |"
        line2 = "| " + " | ".join("---" for _ in headers) + " |"
        body = ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join([line1, line2, *body])

    def render_summary_markdown(self, selected: List[ReportThread]) -> str:

        now = datetime.now(timezone.utc).isoformat()
        out = []
        out.append(f"# Portfolio Health Report\n\nGenerated at: {now}\n")

        # Executive Summary
        out.append("## Executive Summary\n")
        if not selected:
            out.append("_No threads met the attention threshold._\n")
        else:
            rows = []
            for t in selected[:10]:
                when = t.meta.last_seen_utc.isoformat() if t.meta and t.meta.last_seen_utc is not None else "unknown"
                top = max(t.findings, key=lambda x: x.attention_score)
                rows.append([t.thread_id, f"{t.top_score:.2f}", top.flag, top.label, when])
            out.append(self.md_table(["Thread", "Score", "Top Flag", "Label", "Last Seen"], rows))
            out.append("")

        # Section A – Unresolved items
        out.append("## A. Unresolved Items\n")
        rows = []
        for t in selected:
            for f in t.findings:
                if f.flag == "unresolved" and f.label == "unresolved":
                    when = t.meta.last_seen_utc.isoformat() if t.meta and t.meta.last_seen_utc is not None else "unknown"
                    first = t.meta.first_seen_utc.isoformat() if t.meta and t.meta.first_seen_utc is not None else "unknown"
                    rows.append([t.thread_id, f"{f.attention_score:.2f}", f"{f.confidence:.2f}", first, when])
        if rows:
            out.append(self.md_table(["Thread", "Score", "Conf.", "First Seen", "Last Seen"], rows))
        else:
            out.append("_None detected._")
        out.append("")

        # Section B – Risks / Blockers
        out.append("## B. Risks / Blockers\n")
        rows = []
        for t in selected:
            for f in t.findings:
                if f.flag == "risk" and f.label == "risk_present":
                    ev = f.evidence[0].quote if f.evidence else ""
                    rows.append([t.thread_id, f"{f.attention_score:.2f}", f"{f.confidence:.2f}",
                                 ev[:80] + ("…" if len(ev) > 80 else "")])
        if rows:
            out.append(self.md_table(["Thread", "Score", "Conf.", "Key Evidence"], rows))
        else:
            out.append("_None detected._")
        out.append("")

        # Section C – Cost-Critical
        out.append("## C. Cost-Critical Threads\n")
        rows = []
        for t in selected:
            for f in t.findings:
                if f.flag == "cost" and f.label == "cost_involved":
                    ev = f.evidence[0].quote if f.evidence else ""
                    rows.append([t.thread_id, f"{f.confidence:.2f}", ev[:80] + ("…" if len(ev) > 80 else "")])
        if rows:
            out.append(self.md_table(["Thread", "Confidence", "Evidence"], rows))
        else:
            out.append("_None detected._")
        out.append("")

        # Section D – Gross Negligence
        out.append("## D. Ambiguous Ownerships / Accountability \n")
        rows = []
        for t in selected:
            for f in t.findings:
                if f.flag == "ownership" and f.label == "ownership_issue":
                    ev = f.evidence[0].quote if f.evidence else ""
                    rows.append(
                        [t.thread_id, f.label, f"{f.attention_score:.2f}", ev[:80] + ("…" if len(ev) > 80 else "")])
        if rows:
            out.append(self.md_table(["Thread", "Type", "Score", "Evidence"], rows))
        else:
            out.append("_None detected._")
        out.append("")

        return "\n".join(out) + "\n"

    # -------------------------- Normalization -------------------------
    def _normalize_and_score_thread(
        self, thread_id: str, agent_map) -> List[Finding]:
        meta = self._meta.get(str(thread_id))
        out: List[Finding] = []
        for agent_key, payload in agent_map.items():
            if not payload:
                continue
            try:
                finding = self._normalize_one(thread_id, agent_key, payload)
                finding.attention_score = self._score_finding(finding, meta)
                out.append(finding)
            except Exception:
                # Skip malformed payloads; upstream agents should already validate
                continue
        return out

    def _normalize_one(self, thread_id: str, agent_key: str, payload: Dict[str, Any]) -> Finding:
        # Map agent key to flag namespace
        key_map = {
            "unresolved_items": "unresolved",
            "unresolved": "unresolved",
            "risks_blockers": "risk",
            "risk": "risk",
            "cost_considerations": "cost",
            "cost": "cost",
            "possible ownership issue": "ownership",
            "ownership": "ownership",
        }
        flag = key_map.get(agent_key, agent_key)
        label = str(payload.get("label", "")).strip()
        confidence = float(payload.get("confidence", 0.0))
        ev_list = payload.get("evidence", []) or []
        evidence = [Evidence(message_id=str(e.get("message_id", "")), quote=str(e.get("quote", ""))) for e in ev_list]
        return Finding(
            thread_id=str(thread_id),
            flag=flag,
            label=label,
            confidence=confidence,
            evidence=evidence,
        )

    # ---------------------------- Scoring -----------------------------
    def _score_finding(self, f: Finding, meta: Optional[ThreadMeta]) -> float:
        recency = self._recency_weight(meta.last_seen_utc) if meta else 0.0
        if f.flag == "unresolved":
            open_w = self._open_duration_weight(meta.first_seen_utc, meta.last_seen_utc) if meta else 0.0
            severity = 1.0 if f.label == "unresolved" else 0.0  # resolved/no_resolution_needed → 0
            base = 0.6 * f.confidence + 0.2 * recency + 0.2 * open_w
            return base * severity
        if f.flag == "risk":
            severity = 1.0 if f.label == "risk_present" or "blocker_present" else 0.0
            return severity * (0.7 * f.confidence + 0.3 * recency)
        if f.flag == "cost":
            if f.label == "no_cost":
                return 0.0
            severity = 1.0
            return severity * f.confidence
        if f.flag == "ownership":
            if f.label == "none":
                return 0.0
            # High impact by default; recency still matters a bit
            return 0.9 * f.confidence + 0.1 * recency
        return 0.0

    def _recency_weight(self, last_seen) -> float:
        if last_seen is None or pd.isna(last_seen):
            return 0.0
        now = pd.Timestamp.utcnow()  # UTC
        delta_days = (now - last_seen.tz_convert("UTC")).total_seconds() / 86400.0
        w = 1.0 - max(0.0, min(1.0, delta_days / float(self.recency_window_days)))
        return float(w)

    def _open_duration_weight(
        self, first_seen, last_seen
    ) -> float:
        if not first_seen or not last_seen or pd.isna(first_seen) or pd.isna(last_seen):
            return 0.0
        dur_days = (last_seen.tz_convert("UTC") - first_seen.tz_convert("UTC")).total_seconds() / 86400.0
        w = max(0.0, min(1.0, dur_days / float(self.open_window_days)))
        return float(w)

    # participants may be JSON/list/str
    @staticmethod
    def parse_participants(x) -> List[str]:
        if isinstance(x, list):
            return [str(i) for i in x]
        try:
            import json as _json
            parsed = _json.loads(x)
            if isinstance(parsed, list):
                return [str(i) for i in parsed]
        except Exception:
            pass
        s = str(x).strip()
        return [s] if s else []
    # ---------------------------- Metadata ----------------------------
    def _load_thread_meta(self, path: Path) -> None:
        df = pd.read_parquet(path)
        # permissive casts
        df["thread_id"] = df["thread_id"].astype(str)
        if "first_seen_utc" in df.columns:
            df["first_seen_utc"] = pd.to_datetime(df["first_seen_utc"], errors="coerce", utc=True)
        else:
            df["first_seen_utc"] = pd.NaT
        if "last_seen_utc" in df.columns:
            df["last_seen_utc"] = pd.to_datetime(df["last_seen_utc"], errors="coerce", utc=True)
        else:
            df["last_seen_utc"] = pd.NaT


        parts_col = df["participants"] if "participants" in df.columns else [None] * len(df)
        participants = [self.parse_participants(v) for v in parts_col]

        for tid, f, l, p in zip(df["thread_id"], df["first_seen_utc"], df["last_seen_utc"], participants):
            self._meta[str(tid)] = ThreadMeta(
                thread_id=str(tid),
                first_seen_utc=f,
                last_seen_utc=l,
                participants=p,
            )

    # --------------------------- Persistence --------------------------
    @staticmethod
    def write_thread_findings(out_dir: Path, report_threads: List[ReportThread]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for t in report_threads:
            path = out_dir / _safe_name(f"{t.thread_id}.json")
            payload = {
                "thread_id": t.thread_id,
                "top_score": t.top_score,
                "meta": {
                    "first_seen_utc": t.meta.first_seen_utc.isoformat() if t.meta and t.meta.first_seen_utc is not None else None,
                    "last_seen_utc": t.meta.last_seen_utc.isoformat() if t.meta and t.meta.last_seen_utc is not None else None,
                    "participants": t.meta.participants if t.meta else None,
                },
                "findings": [
                    {
                        "flag": f.flag,
                        "label": f.label,
                        "confidence": f.confidence,
                        "attention_score": f.attention_score,
                        "evidence": [asdict(ev) for ev in f.evidence],
                    }
                    for f in t.findings
                ],
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)

    @staticmethod
    def write_summary_md(path: Path, markdown_text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(markdown_text)
