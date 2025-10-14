from __future__ import annotations

# =========================
# Imports & typing
# =========================
import re
import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import pandas as pd
from dateutil import parser as dtparse


# =========================
# Regexes (single place)
# =========================

# Colleagues lines: "Role: Name (email)"
COLLEAGUE_LINE = re.compile(
    r"^\s*(?P<role>[^:]+):\s*(?P<name>.+?)\s*\((?P<email>[^)]+)\)\s*$"
)

# Header block for messages: any run of these headers in any order (per line)
HDR_BLOCK = re.compile(
    r"(?P<hdr>(?:(?:^)(?:From|To|Cc|Date|Subject):[^\n]*\n)+)",
    re.M,
)

# Email extraction (simple & pragmatic)
EMAIL_RE = re.compile(r"[\w\.\-\+]+@[\w\.-]+")

# Forwarded markers, tolerant to dashes and variants: e.g. "--- Forwarded Message ---"
FWD_MARKER = re.compile(
    r"^\s*[-–—]*\s*(?:forwarded|fwd|original)\s+message\s*[:\-–—]*\s*[-–—]*\s*$",
    re.I | re.M,
)

# Single header lines inside forwarded blocks in body
INNER_HDR_LINE = re.compile(r"^(From|To|Cc|Date|Subject):\s*(.+)$", re.M)


# =========================
# Dataclass for normalized messages
# =========================

@dataclass
class ParsedMessage:
    thread_id: str
    email_id: str
    msg_idx: int
    date_local: str
    date_utc: pd.Timestamp
    from_id: str
    to_ids: List[str]
    cc_ids: List[str]
    subject: str
    subject_root: str
    body: str
    message_hash: str


# =========================
# Colleagues / People helpers
# =========================

def load_people(colleagues_txt_path: Path) -> pd.DataFrame:
    """
    Parse colleagues into people_df with deterministic IDs (email-as-ID here).
    Columns: person_id, primary_email, name, role, team, aliases.
    """
    rows = []
    for line in colleagues_txt_path.read_text(encoding="utf-8").splitlines():
        m = COLLEAGUE_LINE.match(line)
        if not m:
            continue
        role = m.group("role").strip()
        name = m.group("name").strip()
        email = m.group("email").strip().lower()
        rows.append(
            {
                "person_id": email,        # stable for now; can swap to UUID later
                "primary_email": email,
                "name": name,
                "role": role,
                "team": None,
                "aliases": [],             # will append human name aliases here
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["primary_email"]).reset_index(drop=True)
    return df


def normalize_name(s: str) -> str:
    """Normalize human name: collapse spaces, NFKC, lowercase."""
    s = " ".join(s.split())
    s = unicodedata.normalize("NFKC", s).strip()
    return s.lower()


def build_name_index(people_df: pd.DataFrame) -> Dict[str, str]:
    """
    name (normalized) -> primary_email
    Used to resolve 'To:' lines that contain only names.
    """
    idx = {}
    for row in people_df.itertuples(index=False):
        if isinstance(row.name, str) and isinstance(row.primary_email, str):
            idx[normalize_name(row.name)] = row.primary_email
    return idx


def add_alias(email: str, alias: str, email_to_idx: Dict[str, int], people_df: pd.DataFrame) -> None:
    """Append a human-readable alias (display name) to a known person."""
    i = email_to_idx.get(email)
    if i is None:
        return
    lst = people_df.at[i, "aliases"]
    if not isinstance(lst, list):
        lst = []
    if alias not in lst:
        lst = lst + [alias]
    people_df.at[i, "aliases"] = lst


# =========================
# Header / Text parsing helpers
# =========================

def grab(hdr_text: str, key: str) -> str:
    """Extract a single header value by key from a header block."""
    m = re.search(rf"^{key}:\s*(.+)$", hdr_text, re.M)
    return m.group(1).strip() if m else ""


def extract_emails(s: Optional[str]) -> List[str]:
    """Extract emails from free text."""
    if not s:
        return []
    return [e.lower() for e in EMAIL_RE.findall(s)]


def normalize_subject_root(subject: str) -> str:
    """Strip leading Re:/Fwd:/Fw: noise for thread grouping."""
    s = subject.strip()
    while True:
        t = re.sub(r"^(re|fw|fwd):\s*", "", s, flags=re.I)
        if t == s:
            break
        s = t
    return s.strip()


def to_utc(date_str: str) -> pd.Timestamp:
    """Parse any date string and convert to UTC (tz-aware)."""
    return pd.Timestamp(dtparse.parse(date_str).astimezone(timezone.utc))


def msg_hash(from_email: str, date_local: str, subject: str, body: str) -> str:
    """Stable dedupe/audit key using a subset of the message content."""
    head = f"{from_email}|{date_local}|{subject}|{body[:200]}".encode("utf-8", errors="ignore")
    return hashlib.sha256(head).hexdigest()


# =========================
# Forwarded-message helpers
# =========================

def subject_is_forward(subject: str) -> bool:
    """Heuristic: subject starts with Fwd:/FW: etc."""
    return bool(re.match(r"^\s*(fwd?|fw)\s*:", subject, flags=re.I))


def extract_forwarded_headers(body: str) -> Optional[Dict[str, str]]:
    """
    Find a 'Forwarded Message' header block *inside* the body (not the main headers)
    and return any inner headers we can capture. Used only for alias harvesting.
    """
    m = FWD_MARKER.search(body)
    if not m:
        return None
    sub = body[m.end():]
    headers: Dict[str, str] = {}
    for key, val in INNER_HDR_LINE.findall(sub):
        if key not in headers:
            headers[key] = val.strip()
    return headers or None


def last_nonempty_line_before(text: str, pos: int) -> str:
    """Return the last non-empty line before absolute position pos ('' if none)."""
    start = max(0, pos - 4000)  # small window is plenty
    chunk = text[start:pos]
    for ln in reversed(chunk.splitlines()):
        s = ln.strip()
        if s:
            return s
    return ""


def is_inner_forwarded_header(text: str, header_match: re.Match) -> bool:
    """
    True if the given header match is the inner block of a forwarded message:
    - directly preceded (ignoring blank lines) by a forwarded marker,
    - often lacking a To: line in your dataset.
    """
    prev_line = last_nonempty_line_before(text, header_match.start())
    if FWD_MARKER.match(prev_line):
        return True
    # Additional guard: if *immediately* after a marker with only whitespace gap.
    last_marker = None
    for mm in FWD_MARKER.finditer(text, 0, header_match.start()):
        last_marker = mm
    if last_marker:
        gap = text[last_marker.end(): header_match.start()]
        if gap.strip() == "":
            return True
    return False


# =========================
# Splitter: thread text -> [(header_match, body)]
# =========================

def split_thread_into_messages(thread_txt: str) -> List[Tuple[re.Match, str]]:
    """
    Split a thread file into message blocks based on header runs.
    We also filter out *inner forwarded* header blocks so they don't become standalone messages.
    """
    parts: List[Tuple[re.Match, str]] = []
    matches = list(HDR_BLOCK.finditer(thread_txt))

    # Filter: remove inner-forwarded header blocks
    filtered: List[re.Match] = []
    for m in matches:
        if is_inner_forwarded_header(thread_txt, m):
            continue
        filtered.append(m)

    # Slice bodies between successive header matches
    for i, m in enumerate(filtered):
        start_body = m.end()
        end_body = filtered[i + 1].start() if i + 1 < len(filtered) else len(thread_txt)
        body = thread_txt[start_body:end_body].strip()
        parts.append((m, body))
    return parts


# =========================
# Recipient resolution
# =========================

def resolve_recipients(line: str, name_to_email: Dict[str, str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Parse a To/Cc line that may contain:
      - "Name (email)"
      - "Name"
      - "email"
    Returns (emails, alias_updates), where alias_updates is [(email, original_display_name)].
    """
    emails: List[str] = []
    alias_updates: List[Tuple[str, str]] = []

    tokens = [t.strip() for t in (line or "").split(",") if t.strip()]
    for tok in tokens:
        found = extract_emails(tok)
        if found:
            e = found[0]
            emails.append(e)
            # If token also contains a display name, record it as alias
            name_part = tok
            for em in found:
                name_part = name_part.replace(em, "").strip(" ()")
            name_part = name_part.strip()
            if name_part:
                alias_updates.append((e, name_part))
        else:
            nm = normalize_name(tok)
            if nm in name_to_email:
                e = name_to_email[nm]
                emails.append(e)
                alias_updates.append((e, tok))

    # de-dup while preserving order
    seen: Set[str] = set()
    deduped: List[str] = []
    for e in emails:
        if e not in seen:
            seen.add(e)
            deduped.append(e)
    return deduped, alias_updates


def thread_key_from_subject(subject_root: str) -> str:
    """
    Deterministic thread id built from subject_root.
    We normalize to lowercase and trim. (You could hash if you prefer.)
    """
    return (subject_root or "").strip().lower()

def parse_forwarded_block(body: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Return (inner_headers, inner_body) from a forwarded block inside 'body'.
    inner_headers: dict like {'From': '...', 'To': '...', 'Date': '...', 'Subject': '...'}
    inner_body: the text AFTER the inner headers (first blank line ends headers).
    If not found, returns (None, None).
    """
    m = FWD_MARKER.search(body)
    if not m:
        return None, None
    sub = body[m.end():]

    # find header lines first
    headers: Dict[str, str] = {}
    header_end = 0
    for mm in INNER_HDR_LINE.finditer(sub):
        key, val = mm.group(1), mm.group(2).strip()
        if key not in headers:
            headers[key] = val
        header_end = mm.end()

    if not headers:
        return None, None

    # inner body starts after the first blank line following the header run
    rest = sub[header_end:]
    # split once at first double newline or end
    split_idx = rest.find("\n\n")
    inner_body = (rest[split_idx + 2:] if split_idx != -1 else rest).strip()
    return headers, inner_body


# =========================
# Ingest one thread file -> messages + thread summary + alias updates
# =========================

def ingest_thread_file(thread_path: Path, people_df: pd.DataFrame) -> Tuple[List[ParsedMessage], List[Tuple[str, str]]]:
    """
    Ingest a single .txt thread file and return:
      - messages: List[ParsedMessage]  (thread_id is derived from subject_root and spans files)
      - alias_updates: List[(email, alias_display_name)]

    Behavior for forwards:
      * Keep the OUTER message (From=forwarder, To/Cc=outer recipients).
      * If a forwarded block exists in the body, also add a SYNTHETIC row representing the ORIGINAL email:
          - from_email  = inner "From"
          - to_emails   = inner "To" (resolved), or fallback to OUTER "From" if missing
          - date_local  = inner "Date" (fallback to outer if missing)
          - subject_root= OUTER subject root  (so it belongs to the same thread as the forward)
          - subject     = OUTER subject (e.g., "Fwd: Re: …") for display consistency
          - body        = inner body only
    """
    # Read file and split into header/body parts
    txt = thread_path.read_text(encoding="utf-8")
    parts = split_thread_into_messages(txt)

    # Identity and resolution indexes
    email2pid = {row.primary_email: row.person_id for row in people_df.itertuples(index=False)}
    name_to_email = build_name_index(people_df)

    # Build temporary rows (dicts) before converting to ParsedMessage
    tmp_rows: List[Dict] = []
    alias_updates_out: List[Tuple[str, str]] = []
    source_email_id = thread_path.stem  # original file stem for traceability

    for m, body in parts:
        # Parse the (outer) header block in any order
        hdr_text  = m.group("hdr") if "hdr" in m.groupdict() else txt[m.start():m.end()]
        from_line = grab(hdr_text, "From")
        to_line   = grab(hdr_text, "To")
        cc_line   = grab(hdr_text, "Cc")
        date_local = grab(hdr_text, "Date")
        subject    = grab(hdr_text, "Subject")

        # Guard: if this header run is actually an inner forwarded header (rare fallback) skip it
        if not to_line.strip() and is_inner_forwarded_header(txt, m):
            continue

        # Resolve OUTER recipients (names → emails where possible)
        outer_to_emails, outer_alias_updates = resolve_recipients(to_line, name_to_email)
        cc_emails, cc_alias_updates = resolve_recipients(cc_line, name_to_email)

        # Sender (outer From)
        from_emails = extract_emails(from_line)
        from_email = from_emails[0] if from_emails else "unknown@unknown"

        # OUTER row (the forward mail itself, or a normal mail)
        outer_root = normalize_subject_root(subject)
        row_outer = {
            "email_id": source_email_id,          # keep source file id
            "from_email": from_email,
            "to_emails": outer_to_emails,
            "cc_emails": cc_emails,
            "date_local": date_local,
            "date_utc": to_utc(date_local),
            "subject": subject,                   # preserve outer subject text
            "subject_root": outer_root,           # used for thread_id
            "body": body,                         # full body (may contain forwarded block)
        }
        tmp_rows.append(row_outer)
        alias_updates_out.extend(outer_alias_updates)
        alias_updates_out.extend(cc_alias_updates)

        # Forwarded ORIGINAL (synthetic row) — only if a forwarded block is present
        inner_headers, inner_body = parse_forwarded_block(body)
        if inner_headers:
            # Inner "From"
            inner_from_line = inner_headers.get("From", "")
            inner_from_emails = extract_emails(inner_from_line)
            inner_from_email = inner_from_emails[0] if inner_from_emails else "unknown@unknown"

            # Inner "To" (resolve names → emails); if missing, fallback to OUTER From
            inner_to_line = inner_headers.get("To", "")
            inner_to_emails, inner_to_alias_updates = resolve_recipients(inner_to_line, name_to_email)
            if not inner_to_emails and from_email:
                inner_to_emails = [from_email]  # fallback target is the forwarder (outer From)
            alias_updates_out.extend(inner_to_alias_updates)

            # Inner date; fallback to outer if missing
            inner_date_local = inner_headers.get("Date", "") or date_local
            inner_date_utc = to_utc(inner_date_local)

            # Integrate into the SAME thread as the forward: use OUTER root
            row_inner = {
                "email_id": source_email_id,          # same source file
                "from_email": inner_from_email,
                "to_emails": inner_to_emails,
                "cc_emails": [],                      # typically not present in forwarded block
                "date_local": inner_date_local,
                "date_utc": inner_date_utc,
                "subject": subject,                   # keep outer subject text for display consistency
                "subject_root": outer_root,           # <-- key: same thread as the forward
                "body": inner_body or "",
            }
            tmp_rows.append(row_inner)

    # Sort rows by UTC time (stable → preserves file order for equal timestamps)
    tmp_rows.sort(key=lambda r: r["date_utc"])

    # Convert to ParsedMessage list
    messages: List[ParsedMessage] = []
    for i, r in enumerate(tmp_rows):
        from_id = email2pid.get(r["from_email"], r["from_email"])
        to_ids  = [email2pid.get(e, e) for e in r["to_emails"]]
        cc_ids  = [email2pid.get(e, e) for e in r["cc_emails"]]

        thread_id = (r["subject_root"] or "").strip().lower()  # deterministic thread key from subject_root

        messages.append(
            ParsedMessage(
                thread_id=thread_id,
                email_id=r["email_id"],               # preserves source file context
                msg_idx=i,
                date_local=r["date_local"],
                date_utc=pd.Timestamp(r["date_utc"]),
                from_id=from_id,
                to_ids=to_ids,
                cc_ids=cc_ids,
                subject=r["subject"],
                subject_root=r["subject_root"],
                body=r["body"],
                message_hash=msg_hash(r["from_email"], r["date_local"], r["subject"], r["body"]),
            )
        )

    # Deduplicate alias updates (if any duplicates occurred)
    if alias_updates_out:
        seen_pairs: Set[Tuple[str, str]] = set()
        alias_updates_out = [(e, a) for (e, a) in alias_updates_out if not ( (e, a) in seen_pairs or seen_pairs.add((e, a)) )]

    return messages, alias_updates_out


def participants_union(group: pd.DataFrame) -> List[str]:
    """Collect unique participant person_ids from from_id, to_ids, cc_ids."""
    s: Set[str] = set(group["from_id"].tolist())
    for col in ("to_ids", "cc_ids"):
        for v in group[col]:
            if isinstance(v, list):
                s.update(v)
    return sorted(s)


# =========================
# Driver: build tables & write parquet
# =========================

def build_tables(threads_dir: Path, colleagues_txt: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end ingestion:
      - load colleagues -> people_df
      - ingest all files -> all messages (each with thread_id from subject_root) + alias updates
      - apply aliases to known people
      - build messages_df
      - build threads_df by grouping across ALL files on thread_id
    """
    people_df = load_people(colleagues_txt)

    # ensure aliases column exists
    if "aliases" not in people_df.columns:
        people_df["aliases"] = [[] for _ in range(len(people_df))]

    # index for in-place alias updates
    email_to_idx = {email: i for i, email in enumerate(people_df["primary_email"])}

    all_messages: List[ParsedMessage] = []
    alias_updates_acc: List[Tuple[str, str]] = []

    # adjust glob pattern if your files use a prefix
    for txt_path in sorted(threads_dir.glob("*.txt")):
        msgs, alias_updates = ingest_thread_file(txt_path, people_df)
        all_messages.extend(msgs)
        alias_updates_acc.extend(alias_updates)

    # apply name->email aliases (only for known colleagues)
    for email, alias in alias_updates_acc:
        add_alias(email, alias, email_to_idx, people_df)

    messages_df = pd.DataFrame(
        [m.__dict__ for m in all_messages],
        columns=[
            "thread_id", "email_id",  # <- email_id added
            "msg_idx", "date_local", "date_utc",
            "from_id", "to_ids", "cc_ids",
            "subject", "subject_root", "body", "message_hash",
        ],
    )

    if not messages_df.empty and not isinstance(messages_df["date_utc"], pd.DatetimeTZDtype):
        messages_df["date_utc"] = pd.to_datetime(messages_df["date_utc"], utc=True)

    # threads_df: group by thread_id (== normalized subject_root)
    threads_rows: List[Dict] = []
    if not messages_df.empty:
        for tid, g in messages_df.groupby("thread_id", sort=True):
            threads_rows.append({
                "thread_id": tid,
                "subject_root": next((x for x in g["subject_root"].tolist() if x), ""),
                "first_seen_utc": g["date_utc"].min(),
                "last_seen_utc": g["date_utc"].max(),
                "participants": participants_union(g),
                "message_count": int(len(g)),
            })

    threads_df = pd.DataFrame(
        threads_rows,
        columns=["thread_id", "subject_root", "first_seen_utc", "last_seen_utc", "participants", "message_count"],
    )
    if not threads_df.empty and not isinstance(threads_df["first_seen_utc"], pd.DatetimeTZDtype):
        threads_df["first_seen_utc"] = pd.to_datetime(threads_df["first_seen_utc"], utc=True)
        threads_df["last_seen_utc"]  = pd.to_datetime(threads_df["last_seen_utc"],  utc=True)

    return people_df, messages_df, threads_df



def write_parquet(out_dir: Path, people_df: pd.DataFrame, messages_df: pd.DataFrame, threads_df: pd.DataFrame) -> None:
    """Persist the three tables as Parquet (columnar, compressed, analytics-friendly)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    people_df.to_parquet(out_dir / "people.parquet", index=False)
    messages_df.to_parquet(out_dir / "messages.parquet", index=False)
    threads_df.to_parquet(out_dir / "threads.parquet", index=False)


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    base = Path(r"E:\Work\Atrecto\business_report_system")
    threads_dir = base / "data"
    colleagues  = base / "data" / "Colleagues.txt"
    out_dir     = base / "warehouse"

    people_df, messages_df, threads_df = build_tables(threads_dir, colleagues)
    write_parquet(out_dir, people_df, messages_df, threads_df)
    print("Wrote:", out_dir / "people.parquet", out_dir / "messages.parquet", out_dir / "threads.parquet")
