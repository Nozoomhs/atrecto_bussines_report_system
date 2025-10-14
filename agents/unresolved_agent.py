from __future__ import annotations

"""
UnresolvedAgent
----------------
Consumes a prompt produced by PromptParser for the "unresolved_items" task and
returns a strict JSON dict that matches the schema expected by your pipeline.

Design:
- Backend-agnostic: pass any client with a `.complete(prompt: str) -> str` method.
  Sample backends provided for OpenAI (API) and Ollama (local, free).
- Schema-enforced with Pydantic and light business rules (label set, reason length,
  evidence message_id pattern, non-empty evidence).
- Auto-retry: if the model returns invalid JSON or violates schema, the agent asks the
  model to correct itself up to `max_retries` times by feeding the validation errors.

Usage:
    agent = UnresolvedAgent(backend=OpenAIChatClient(model="gpt-4o-mini"))
    result = agent.run(prompt_text)

    # Or with Ollama (local):
    agent = UnresolvedAgent(backend=OllamaChatClient(model="qwen2.5:7b-instruct"))
    result = agent.run(prompt_text)

This module does not import heavy libs unless you actually use that backend.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal
from clients.backend import LLMBackend
import json
from pydantic import BaseModel, field_validator
from clients.ollama_chat_client import OllamaChatClient
from clients.openai_chat_client import OpenAIChatClient


# ======================= Schema =======================

ALLOWED_LABELS = {"resolved", "unresolved", "no_resolution_needed"}


class EvidenceItem(BaseModel):
    message_id: str
    quote: str
    @field_validator("quote")
    def quote_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("quote must be non-empty")
        return v.strip()

class UnresolvedOutput(BaseModel):
    label: str
    reason_short: str
    evidence: List[EvidenceItem]
    confidence: float

    @field_validator("label")
    def label_allowed(cls, v: str) -> str:
        if v not in ALLOWED_LABELS:
            raise ValueError(f"label must be one of {sorted(ALLOWED_LABELS)}")
        return v

    @field_validator("reason_short")
    def reason_len(cls, v: str) -> str:
        if len(v.strip()) == 0:
            raise ValueError("reason_short must be non-empty")
        if len(v) > 100:
            raise ValueError("reason_short must be ≤ 50 chars")
        return v

    @field_validator("confidence")
    def score_range(cls, v: float) -> float:
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError("scores must be between 0.0 and 1.0")
        return float(v)

    @field_validator("evidence")
    def evidence_nonempty(cls, v: List[EvidenceItem]) -> List[EvidenceItem]:
        if not v:
            raise ValueError("evidence must contain at least one item")
        return v

# ======================= Agent =======================

class UnresolvedAgent:
    def __init__(self, backend: LLMBackend, max_retries: int = 2) -> None:
        self.backend = backend
        self.max_retries = max_retries

    def run(self, prompt: str) -> Dict[str, Any]:
        """Run the agent, validate JSON, retry on violations. Returns a Python dict.
        Raises ValueError if still invalid after retries.
        """
        last_err = None
        content = self.backend.complete(prompt, format="json")
        for attempt in range(self.max_retries + 1):
            try:
                payload = self._coerce_to_json(content)
                obj = UnresolvedOutput.model_validate(payload)
                return obj.model_dump()
            except Exception as e:
                last_err = str(e)
                # Ask model to fix its output using the error message
                content = self._repair(prompt, content, last_err)
        raise ValueError(f"Model could not produce valid JSON after retries: {last_err}")

    # -------------------- helpers --------------------
    @staticmethod
    def _coerce_to_json(text: str) -> Dict[str, Any]:
        if not text:
            raise ValueError("empty model output")
        s = text.strip()
        # Strip code fences if present
        if s.startswith("```"):
            s = s.strip("`\n ")
            # possible language label at start
            if s.startswith("json"):
                s = s[4:].lstrip("\n")
        # Find first and last braces to recover a JSON object if extra tokens exist
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object detected")
        js = s[start : end + 1]
        return json.loads(js)

    def _repair(self, original_prompt: str, bad_output: str, error_msg: str) -> str:
        fix_prompt = (
            "You produced invalid JSON. Here is the error message; fix ONLY the JSON and return a valid object that meets the schema.\n"
            f"Validation error: {error_msg}\n"
            "Your previous output was:\n" + bad_output + "\n"
            "Return ONLY JSON."
        )
        merged = original_prompt + "\n\n" + fix_prompt
        return self.backend.complete(merged, format="json")


# -------------------- Optional quick test --------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run UnresolvedAgent on a prebuilt prompt (stdin or file)")
    ap.add_argument("--from_file", type=str, default=None)
    ap.add_argument("--backend", choices=["openai", "ollama"], default="ollama")
    ap.add_argument("--model", type=str, default=None, help="Override model name for the chosen backend")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    # Read input JSON (from file or stdin) and extract only the "prompt" field
    if args.from_file:
        with open(args.from_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

    prompt_text = payload.get("prompt")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("Input JSON must contain a non-empty 'prompt' field")

    if args.backend == "openai":
        model = args.model or "gpt-4o-mini"
        backend = OpenAIChatClient(model=model)
    else:
        model = args.model or "qwen2.5:7b-instruct"
        backend = OllamaChatClient(model=model)

    agent = UnresolvedAgent(backend=backend, max_retries=args.retries)
    out = agent.run(prompt_text)
    print(json.dumps(out, ensure_ascii=False, indent=2))
