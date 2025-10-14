from __future__ import annotations
"""
run_agents_and_aggregate.py
---------------------------
Wrapper that:
1) Scans prompt folders for four agents
2) Runs each agent (Ollama or OpenAI backends)
3) Gathers outputs per thread
4) Uses FinalReportAgent to score, select, and write findings + summary
5) Generates Human readable report

Expected prompt layout (from your parser):
  prompts/
    unresolved_agent/<thread_id>.json
    risk_agent/<thread_id>.json
    cost_agent/<thread_id>.json
    ownership_agent/<thread_id>.json

Outputs (default):
  outputs/
    agent_results/<agent>/<thread_id>.json     
    findings/<thread_id>.json                 
    summary/summary.md  
    summary/executive_summary.md                       
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List
import argparse
import re

# Agents
from agents.unresolved_agent import UnresolvedAgent
from agents.risk_agent import RiskAgent
from agents.cost_agent import CostAgent
from agents.ownership_agent import OwnershipAgent
from tools.string_tools import _safe_name
# Backends
from clients.ollama_chat_client import OllamaChatClient
from clients.openai_chat_client import OpenAIChatClient
from clients.backend import LLMBackend

# Aggregator
from agents.final_report_agent import FinalReportParser
# Huma Report
from agents.human_report_agent import HumanReportAgent

AGENT_FOLDERS = {
    "unresolved": "unresolved_agent",
    "risk": "risk_agent",
    "cost": "cost_agent",
    "ownership": "ownership_agent",
}

AGENT_CLASSES = {
    "unresolved": UnresolvedAgent,
    "risk": RiskAgent,
    "cost": CostAgent,
    "ownership": OwnershipAgent,
}


def load_prompt_and_tid(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompt = payload.get("prompt")
    tid = str(payload.get("thread_id"))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Bad prompt JSON at {path}: missing 'prompt'")
    if not tid or tid.lower() in {"none", "nan"}:
        # fallback to filename if really missing
        tid = path.stem
    return prompt, tid


def run_agent(agent_key: str, agent, prompt_text: str) -> Dict[str, Any]:
    try:
        return agent.run(prompt_text)
    except Exception as e:
        # Return a minimal failure record; aggregator will ignore missing/invalid
        return {"_error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run four agents over prompts and aggregate findings")
    ap.add_argument("--prompts_dir", default="prompts", type=str)
    ap.add_argument("--messages_parquet", type=str, default="messages.parquet")
    ap.add_argument("--people_parquet", type=str, default="people.parquet")
    ap.add_argument("--out_dir", default="outputs", type=str)
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--threads_parquet", type=str, default="threads.parquet")
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    prompts_dir = Path(args.prompts_dir)
    out_dir = Path(args.out_dir)
    agent_out_dir = out_dir / "agent_results"
    findings_dir = out_dir / "findings"
    summary_path = out_dir / "summary" / "summary.md"
    warehouse_path = Path(os.getcwd()) / "warehouse"

    # Backend
    # Currently only Ollama is tested, Openai requires billing setup.
    if args.backend == "openai":
        model = args.model or "gpt-4o-mini"
        backend: LLMBackend = OpenAIChatClient(model=model)
    else:
        model = args.model or "qwen2.5:7b-instruct"
        backend = OllamaChatClient(model=model)

    # Discover (original) thread IDs by reading each prompt JSON
    thread_ids: set[str] = set()
    prompt_files: dict[str, dict[str, Path]] = {k: {} for k in AGENT_FOLDERS}  # {agent_key: {tid: path}}

    for agent_key, folder in AGENT_FOLDERS.items():
        ag_dir = prompts_dir / folder
        if not ag_dir.exists():
            continue
        for fp in ag_dir.glob("*.json"):
            _, tid = load_prompt_and_tid(fp)
            thread_ids.add(tid)
            prompt_files[agent_key][tid] = fp

    # Instantiate agents
    agents = {
        k: AGENT_CLASSES[k](backend=backend, max_retries=args.max_retries) for k in AGENT_CLASSES
    }

    # Run agents per thread
    per_thread_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for tid in sorted(thread_ids):
        per_thread_results[tid] = {}
        for agent_key, folder in AGENT_FOLDERS.items():
            fp = prompt_files.get(agent_key, {}).get(tid)
            if not fp:
                continue
            prompt_text, original_tid = load_prompt_and_tid(fp)  # original_tid == tid

            result = run_agent(agent_key, agents[agent_key], prompt_text)

            # Save raw agent output (use safe filename but keep original tid in memory)
            out_sub = agent_out_dir / folder
            out_sub.mkdir(parents=True, exist_ok=True)
            with open(out_sub / f"{_safe_name(original_tid)}.json", "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)

            if isinstance(result, dict) and "_error" not in result:
                per_thread_results[tid][agent_key] = result

    # Aggregate
    aggregator = FinalReportParser(
        threads_parquet=warehouse_path / args.threads_parquet  if args.threads_parquet else None,
        messages_parquet=warehouse_path / args.messages_parquet if args.messages_parquet else None,
        people_parquet=warehouse_path / args.people_parquet if args.people_parquet else None,
        threshold=args.threshold,
        top_k=args.top_k,
    )
    selected, all_threads = aggregator.build_report(per_thread_results)

    # Persist
    aggregator.write_thread_findings(findings_dir, selected)
    summary_md = aggregator.render_summary_markdown(selected)
    aggregator.write_summary_md(summary_path, summary_md)

    try:
        exec_summary_path = out_dir / "summary" / "executive_summary.md"
        hr = HumanReportAgent(backend=backend)
        executive_md = hr.generate_from_text(
            summary_md,
            report_title="QBR Executive Summary",
        )
        exec_summary_path.parent.mkdir(parents=True, exist_ok=True)
        exec_summary_path.write_text(executive_md, encoding="utf-8")
    except Exception as _e:
        # Non-fatal: keep the machine summary if LLM summarization fails
        pass

    # Also write a compact index of all threads with top_score
    index = [
        {"thread_id": t.thread_id, "top_score": t.top_score} for t in
        sorted(all_threads, key=lambda x: x.top_score, reverse=True)
    ]
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary" / "index.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {len(selected)} selected threads → {summary_path}")
    print(f"Wrote executive summary → {exec_summary_path}")

if __name__ == "__main__":
    main()
