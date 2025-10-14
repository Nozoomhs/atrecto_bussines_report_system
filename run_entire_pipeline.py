import os
from pathlib import Path
import argparse
import re
import json
from typing import Dict, Any

from parsing import build_tables, write_parquet
from prompt_parser import PromptParser
from run_agents_and_aggregate import load_prompt_and_tid, run_agent
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

if __name__ == "__main__":
    # possible arguments
    ap = argparse.ArgumentParser(description="Build per-thread prompts for analytical agents.")
    ap.add_argument("--threads",  help="Path to threads.parquet")
    ap.add_argument("--messages",  help="Path to messages.parquet")
    ap.add_argument("--people",  help="Path to people.parquet")
    ap.add_argument("--tz", default="Europe/Budapest", help="Timezone for rendering timestamps")
    ap.add_argument("--max_messages", type=int, default=None, help="Optional cap of messages per thread in prompts")
    ap.add_argument("--sample", type=int, default=0, help="Print prompts for N threads and exit")
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    # --- Parquet Parsing ---
    base = Path(os.getcwd())
    threads_dir = base / "data"
    colleagues  = base / "data" / "Colleagues.txt"
    out_dir     = base / "warehouse"

    people_df, messages_df, threads_df = build_tables(threads_dir, colleagues)
    write_parquet(out_dir, people_df, messages_df, threads_df)
    print("Wrote:", out_dir / "people.parquet", out_dir / "messages.parquet", out_dir / "threads.parquet")

    # --- Prompt Creation ---
    if args.messages is None:
        args.messages = out_dir / "messages.parquet"
    if args.people is None:
        args.people = out_dir / "people.parquet"
    if args.threads is None:
        args.threads = out_dir / "threads.parquet"

    parser = PromptParser(args.threads, args.messages, args.people, tz=args.tz, max_messages=args.max_messages)
    prompts = parser.build_all_prompts()
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
    print("wrote prompts to", base_dir)



    ##  -- Run the agents and aggregation --
    prompts_dir = Path("prompts")
    out_dir = Path("outputs")
    agent_out_dir = out_dir / "agent_results"
    findings_dir = out_dir / "findings"
    summary_path = out_dir / "summary" / "summary.md"
    warehouse_path = Path(os.getcwd()) / "warehouse"

    # Backend
    # Currently only Ollama is tested, Openai requires billing setup.
    # if args.backend == "openai":
    #     model = args.model or "gpt-4o-mini"
    #     backend: LLMBackend = OpenAIChatClient(model=model)
    # else:
    model = "qwen2.5:7b-instruct"
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
        threads_parquet=warehouse_path ,
        messages_parquet=warehouse_path ,
        people_parquet=warehouse_path ,
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
        print(_e)
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
