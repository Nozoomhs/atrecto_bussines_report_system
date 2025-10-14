from __future__ import annotations

"""
PromptParser: loads Threads.parquet, Messages.parquet, People.parquet and emits
per-thread prompts tailored for four downstream agents:
  - cost_considerations
  - unresolved_items
  - risks_blockers
  - ownership


Assumptions (safe defaults for POC):
- Parquet schemas (flexible, handled defensively):
  * Threads.parquet: columns include [thread_id, first_seen_utc, last_seen_utc, participants]
  * Messages.parquet: [thread_id, msg_idx (int, chronological), utc (timestamp or str),
                       from_id (email), to_ids (list|str), cc_ids (list|str), body (str), message_hash]
  * People.parquet: [person_id (email), name (str|None), role (str|None), aliases (list|str|None)]
- Times are rendered in a chosen timezone (default 'Europe/Budapest')
- Message IDs in prompts use the canonical form: "T{thread_id}/M{msg_idx}".

Usage (example):
    parser = PromptParser(
        threads_path="/path/Threads.parquet",
        messages_path="/path/Messages.parquet",
        people_path="/path/people.parquet",
    )
    prompts_by_thread = parser.build_all_prompts()
    # prompts_by_thread[thread_id]["cost_considerations"] -> str
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
import re
from textwrap import dedent
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Person:
    person_id: str  # canonical email/address key
    name: str
    role: str


class PromptParser:
    def __init__(
        self,
        threads_path: str | Path,
        messages_path: str | Path,
        people_path: str | Path,
        tz: str = "Europe/Budapest",
        max_messages: Optional[int] = None,
    ) -> None:
        """
        Args:
            threads_path: Path to Threads.parquet
            messages_path: Path to Messages.parquet
            people_path: Path to people.parquet
            tz: IANA timezone for rendering timestamps
            max_messages: optional hard cap of messages included per thread (POC safety)
        """
        self.threads_path = Path(threads_path)
        self.messages_path = Path(messages_path)
        self.people_path = Path(people_path)
        self.tz = ZoneInfo(tz)
        self.max_messages = max_messages

        self._threads_df: Optional[pd.DataFrame] = None
        self._messages_df: Optional[pd.DataFrame] = None
        self._people_df: Optional[pd.DataFrame] = None

        # alias -> canonical person_id mapping
        self._alias_map: Dict[str, str] = {}
        # canonical person_id -> Person
        self._people_map: Dict[str, Person] = {}

        # Pre-compiled simple cleaners
        self._sig_re = re.compile(r"(^--+.*$)|(^Sent from my.*$)|(^Best,\s*$)", re.I | re.M)

    # ---------------------- public API ----------------------
    def build_all_prompts(self) -> Dict[str, Dict[str, str]]:
        """Return {thread_id: {agent_key: prompt_text}} for all threads.

        agent_key in {"cost_considerations", "unresolved_items", "risks_blockers", "ownership_issue"}
        """
        self._lazy_load()

        out: Dict[str, Dict[str, str]] = {}
        for thread_id, tdf in self._iter_threads():
            participants_block = self._format_participants_block(thread_id, tdf)
            messages_block = self._format_messages_block(tdf)

            out[thread_id] = {
                "cost_considerations": self._render_cost_prompt(participants_block, messages_block),
                "unresolved_items": self._render_unresolved_prompt(participants_block, messages_block),
                "risks_blockers": self._render_risks_prompt(participants_block, messages_block),
                "ownership_issue": self._render_ownsership_prompt(participants_block, messages_block),
            }
        return out

    def build_prompts_for_thread(self, thread_id: str | int) -> Dict[str, str]:
        """Return {agent_key: prompt_text} for a single thread_id."""
        self._lazy_load()
        thread_id = str(thread_id)
        tdf = self._messages_df[self._messages_df["thread_id"].astype(str) == thread_id]
        if tdf.empty:
            raise KeyError(f"thread_id={thread_id} not found in Messages.parquet")
        tdf = self._prepare_thread_messages(tdf)
        participants_block = self._format_participants_block(thread_id, tdf)
        messages_block = self._format_messages_block(tdf)

        return {
            "cost_considerations": self._render_cost_prompt(participants_block, messages_block),
            "unresolved_items": self._render_unresolved_prompt(participants_block, messages_block),
            "risks_blockers": self._render_risks_prompt(participants_block, messages_block),
            "gross_negligence": self._render_ownsership_prompt(participants_block, messages_block),
        }

    # loading & normalization
    def _lazy_load(self) -> None:
        if self._threads_df is None:
            self._threads_df = pd.read_parquet(self.threads_path)
        if self._messages_df is None:
            self._messages_df = pd.read_parquet(self.messages_path)
        if self._people_df is None:
            self._people_df = pd.read_parquet(self.people_path)
        self._normalize_people()
        # Basic hygiene on messages
        self._messages_df = self._messages_df.copy()
        # Ensure msg_idx is int sortable
        if "msg_idx" in self._messages_df.columns:
            self._messages_df["msg_idx"] = pd.to_numeric(self._messages_df["msg_idx"], errors="coerce").fillna(0).astype(int)

    def _normalize_people(self) -> None:
        """
        Build alias and people maps.
        - If 'aliases' column contains strings, try to parse JSON or split by comma/semicolon.
        - Missing name/role are filled with fallbacks.
        """
        df = self._people_df.copy()
        df["person_id"] = df["person_id"].astype(str)
        df["name"] = df.get("name", pd.Series([None]*len(df))).fillna("")
        df["role"] = df.get("role", pd.Series([None]*len(df))).fillna("Unknown")

        def parse_aliases(x) -> List[str]:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return []
            if isinstance(x, list):
                return [str(a).strip() for a in x if a]
            s = str(x).strip()
            if not s:
                return []
            # try JSON list first
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(a).strip() for a in parsed if a]
            except Exception:
                pass
            # fallback: split by common delimiters
            return [t.strip() for t in re.split(r"[,;\s]+", s) if t.strip()]


        alias_map: Dict[str, str] = {}
        people_map: Dict[str, Person] = {}
        for _, row in df.iterrows():
            pid = row["person_id"].lower()
            person = Person(person_id=pid, name=(row["name"] or pid), role=row["role"])
            people_map[pid] = person
            alias_map[pid] = pid
            for alias in row["aliases"]:
                alias_map[str(alias).lower()] = pid
        self._alias_map = alias_map
        self._people_map = people_map

    # ----- per-thread formatting ----------
    def _iter_threads(self):
        msgs = self._messages_df
        for thread_id, tdf in msgs.groupby(msgs["thread_id"].astype(str)):
            yield thread_id, self._prepare_thread_messages(tdf)

    def _prepare_thread_messages(self, tdf: pd.DataFrame) -> pd.DataFrame:
        tdf = tdf.copy()
        # Sort chronologically by msg_idx then utc
        sort_cols = [c for c in ["msg_idx", "utc"] if c in tdf.columns]
        tdf = tdf.sort_values(sort_cols).reset_index(drop=True)


        # Resolve identities
        tdf["from_person"] = tdf["from_id"].map(self._resolve_person)
        tdf["to_people"] = tdf["to_ids"].map(lambda lst: [self._resolve_person(x) for x in lst])
        tdf["cc_people"] = tdf["cc_ids"].map(lambda lst: [self._resolve_person(x) for x in lst])

        # Clean body a bit (optional, minimal)
        tdf["body_clean"] = tdf.get("body", "").map(self._clean_body)

        # Cap messages if requested (POC guard)
        if self.max_messages is not None and len(tdf) > self.max_messages:
            tdf = tdf.tail(self.max_messages).reset_index(drop=True)
        return tdf

    def _resolve_person(self, pid: str | None) -> Person:
        if not pid:
            return Person(person_id="unknown", name="Unknown", role="Unknown")
        key = str(pid).lower()
        canon = self._alias_map.get(key, key)
        return self._people_map.get(
            canon,
            Person(person_id=canon, name=canon, role="Unknown"),
        )

    def _clean_body(self, text: str | None) -> str:
        if not text:
            return ""
        # light trimming; keep content intact
        s = str(text).strip()
        # remove common signature markers (very conservative)
        s = self._sig_re.sub("", s)
        return s.strip()

    def _format_participants_block(self, thread_id: str, tdf: pd.DataFrame) -> str:
        # Prefer Threads.parquet participants if present
        names_roles: List[Tuple[str, str]] = []
        if self._threads_df is not None and "participants" in self._threads_df.columns:
            row = self._threads_df[self._threads_df["thread_id"].astype(str) == str(thread_id)]
            if not row.empty:
                candidates = row.iloc[0]["participants"] # may be emails or names
                # Resolve to Person if possible
                for c in candidates:
                    p = self._resolve_person(c)
                    names_roles.append((p.name, p.role))
        if not names_roles:
            # derive from messages
            people: Dict[str, Person] = {}
            for _, m in tdf.iterrows():
                for p in [m["from_person"], *m["to_people"], *m["cc_people"]]:
                    people[p.person_id] = p
            names_roles = sorted({(p.name, p.role) for p in people.values()}, key=lambda x: (x[1], x[0]))

        lines = [f"{name} - {role}" for name, role in names_roles]
        return "\n".join(lines)

    def _format_messages_block(self, tdf: pd.DataFrame) -> str:
        lines: List[str] = []
        for i, m in enumerate(tdf.itertuples(index=False), start=1):
            msg_id = getattr(m, "message_hash", None)
            # render time in configured TZ
            utc_ts = getattr(m, "date_utc", None)
            if pd.isna(utc_ts) or utc_ts is None:
                time_s = "unknown"
            else:
                # utc_ts is pandas.Timestamp with tzinfo UTC
                try:
                    time_s = utc_ts.tz_convert(self.tz).isoformat()
                except Exception:
                    # if tz-naive, localize first
                    time_s = pd.to_datetime(utc_ts, utc=True).tz_convert(self.tz).isoformat()
            from_p: Person = getattr(m, "from_person")
            to_ps: List[Person] = list(getattr(m, "to_people"))
            cc_ps: List[Person] = list(getattr(m, "cc_people"))

            to_str = ", ".join(p.name for p in to_ps) if to_ps else "(no direct recipients)"
            cc_str = ", ".join(p.name for p in cc_ps)
            cc_clause = f", cc {cc_str}" if cc_str else ""

            body = getattr(m, "body_clean", "")

            lines.append(
                dedent(
                    f"""
                    Message {i}:
                    {from_p.name} - {from_p.role} sent at {time_s} to {to_str}{cc_clause} with message id {msg_id}
                    {body}
                    """
                ).strip()
            )
        return "\n\n".join(lines)

    # --- agent prompt renderers ---
    def _render_cost_prompt(self, participants_block: str, messages_block: str) -> str:
        template = dedent(
            f"""
            Role:
            You are a supervisor who is going to browse email conversations and look for cost critical resolutions of a problem.
            Your key objective is to identify the possible costs of a solution/plan described in a series of emails from the perspective of the company. 
            You must also reason for your choice and specify if the provided solution will cost the company other than obvious development costs.
            If there is no indication to costly operations in the thread, simply say classify as no_cost.
            Output format (strict JSON):
            {{
              "flag": "cost considerations",
              "label": "no_cost|cost",
              "reason_short": "\u226450 chars",
              "evidence": [{{"message_id": "hash_number", "quote": "…"}}],
              "confidence": 0.0-1.0,
            }}

            Participants in the thread:
            {participants_block}

            Conversation:
            {messages_block}

            Return ONLY JSON.
            """
        ).strip()
        return template

    def _render_unresolved_prompt(self, participants_block: str, messages_block: str) -> str:
        template = dedent(
            f"""
            Role:
            You are an agent that filters threads by resolution status. Decide whether the thread ends in a clear outcome.
            If no resolution is needed (FYI/announcements), choose 'no_resolution_needed'. 
            If a task/problem remains open, choose 'unresolved'. 
            If the outcome is clear (done/shipped/decided), choose 'resolved'.
            State how confident your are in this decision.

            Output format (strict JSON):
            {{
              "flag": "resolution status",
              "label": "resolved|unresolved|no_resolution_needed",
              "reason_short": "\u226450 chars",
              "evidence": [{{"message_id": "hash_number", "quote": "…"}}],
              "confidence": 0.0-1.0,
            }}

            Participants in the thread:
            {participants_block}

            Conversation:
            {messages_block}

            Return ONLY JSON.
            """
        ).strip()
        return template

    def _render_risks_prompt(self, participants_block: str, messages_block: str) -> str:
        template = dedent(
            f"""
            Role:
            You are an agent that detects blockers and risks in a series of emails from the perspective of the company.
            Identify whether the thread indicates a risk or a blocking dependency without a clear mitigation.
            If you can't find evidence for a blocking dependency or a possible risk, simply label it none.
            State how confident your are in this decision. 
            Output format (strict JSON):
            {{
              "flag": "risks/blockers",
              "label": "risk_present|blocker_present|none",
              "risk_type": "dependency|security|legal|operational|timeline|unknown",
              "reason_short": "\u226450 chars",
              "evidence": [{{"message_id": "hash_number", "quote": "…"}}],
              "confidence": 0.0-1.0,
            }}

            Participants in the thread:
            {participants_block}

            Conversation:
            {messages_block}

            Return ONLY JSON.
            """
        ).strip()
        return template


    def _render_ownsership_prompt(self, participants_block: str, messages_block: str) -> str:
        template = dedent(
            f"""
            Role:
            You are an expert agent that identifies Accountability and Ownership problems from series of emails.
            Look for cases where the threads do not end with clear assignments to individuals.
            Only flag threads if there is a clear indication that there is something to be done and the work item is not cleary assigned to someone.
            ---
            Here are two examples:
            
            **Example 1 (Positive Match):**
            Conversation:
            - Sanyi (msg_1): "The login service is down. We need to get this fixed ASAP."
            - Bela (msg_2): "I've confirmed the outage. It seems to be a database connection issue. Someone should look into it."
            - Sunny (msg_3): "Okay, thanks for the update. Let's keep an eye on it."
            Output:
            {{
              "flag": "possible ownership issue",
              "label": "ownership_issue",
              "reason_short": "Critical outage mentioned but no one was assigned to fix it.",
              "evidence": [{{"message_id": "msg_2", "quote": "Someone should look into it."}}],
              "confidence": 0.9
            }}
            
            **Example 2 (Negative Match):**
            Conversation:
            - Raul (msg_1): "Can we get the new banner image for the homepage?"
            - Dominik (msg_2): "Yep, I'm on it. I'll have a draft ready by EOD tomorrow."
            Output:
            {{
              "flag": "possible ownership issue",
              "label": "none",
              "reason_short": "Task was clearly assigned and acknowledged by Dominik.",
              "evidence": [{{"message_id": "msg_2", "quote": "Yep, I'm on it."}}],
              "confidence": 0.95
            }}
            ---
            
            Output format (strict JSON):
            {{
              "flag": "possible ownership issue",
              "label": "none|ownership_issue",
              "reason_short": "\u226450 chars",
              "evidence": [{{"message_id": "hash_number", "quote": "…"}}],
              "confidence": 0.0-1.0,
            }}

            Participants in the thread:
            {participants_block}

            Conversation:
            {messages_block}

            Return ONLY JSON.
            """
        ).strip()
        return template


# ---------------------- optional CLI exercise ----------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build per-thread prompts for analytical agents.")
    ap.add_argument("--threads", required=True, help="Path to threads.parquet")
    ap.add_argument("--messages", required=True, help="Path to messages.parquet")
    ap.add_argument("--people", required=True, help="Path to people.parquet")
    ap.add_argument("--tz", default="Europe/Budapest", help="Timezone for rendering timestamps")
    ap.add_argument("--max_messages", type=int, default=None, help="Optional cap of messages per thread in prompts")
    ap.add_argument("--sample", type=int, default=0, help="Print prompts for N threads and exit")
    args = ap.parse_args()

    parser = PromptParser(args.threads, args.messages, args.people, tz=args.tz, max_messages=args.max_messages)
    prompts = parser.build_all_prompts()

    if args.sample:
        for i, (tid, agent_map) in enumerate(prompts.items()):
            print("#" * 80)
            print(f"THREAD {tid}")
            for agent, prompt in agent_map.items():
                print("\n=== ", agent, " ===\n")
                print(prompt)
                print("\n")
            if i + 1 >= args.sample:
                break
    else:
        # Write each JSON to prompts/<agent_folder>/<thread_id>.json
        base_dir = Path("prompts")
        agent_folder = {
            "cost_considerations": "cost_agent",
            "unresolved_items": "unresolved_agent",
            "risks_blockers": "risk_agent",
            "ownership_issue": "ownership_agent",
        }
        for tid, agent_map in prompts.items():
            safe_tid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(tid))
            for agent_key, prompt in agent_map.items():
                subdir = base_dir / agent_folder.get(agent_key, agent_key)
                subdir.mkdir(parents=True, exist_ok=True)
                rec = {"thread_id": str(tid), "agent": agent_key, "prompt": prompt}
                out_path = subdir / f"{safe_tid}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
