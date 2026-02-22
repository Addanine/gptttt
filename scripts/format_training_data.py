from __future__ import annotations

import argparse
import html as html_mod
import json
import re
from pathlib import Path

QUOTE_RE = re.compile(r">>(\d+)")
MAX_CHAIN_DEPTH = 6
MIN_TEXT_LENGTH = 5

SYSTEM_PROMPT = (
    "You are an anonymous poster on /lgbt/. "
    "Reply to the post naturally, matching the tone and style of the board."
)


def html_to_text(raw: str | None) -> str:
    if not raw:
        return ""
    text = raw.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"<[^>]+>", "", text)
    return html_mod.unescape(text)


def get_text(record: dict) -> str:
    post = record.get("post", {})
    sub = (post.get("sub") or "").strip()
    com_text = (record.get("com_text") or html_to_text(post.get("com"))).strip()
    parts = [p for p in [sub, com_text] if p]
    return "\n\n".join(parts)


def extract_quotes(text: str) -> list[int]:
    return [int(m.group(1)) for m in QUOTE_RE.finditer(text)]


def build_chain(
    post_no: int, posts: dict[int, dict], max_depth: int = MAX_CHAIN_DEPTH
) -> list[dict]:
    chain: list[dict] = []
    current: int | None = post_no
    visited: set[int] = set()

    while current is not None and len(chain) < max_depth and current not in visited:
        visited.add(current)
        record = posts.get(current)
        if not record:
            break
        chain.append(record)

        text = get_text(record)
        quotes = extract_quotes(text)
        resto = int(record.get("post", {}).get("resto", 0))

        parent: int | None = None
        for q in quotes:
            if q != current and q in posts:
                parent = q
                break
        if parent is None and resto and resto != current and resto in posts:
            parent = resto

        current = parent

    chain.reverse()
    return chain


def chain_to_messages(chain: list[dict]) -> list[dict] | None:
    if len(chain) < 2:
        return None

    if len(chain) % 2 != 0:
        chain = chain[1:]

    if len(chain) < 2:
        return None

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, record in enumerate(chain):
        role = "user" if i % 2 == 0 else "assistant"
        text = get_text(record)
        if not text:
            return None
        messages.append({"role": role, "content": text})

    if messages[-1]["role"] != "assistant":
        return None

    return messages


def multi_quote_to_messages(
    reply_record: dict, quoted_records: list[dict]
) -> list[dict] | None:
    reply_text = get_text(reply_record)
    if not reply_text:
        return None

    parts: list[str] = []
    for rec in quoted_records:
        pno = int(rec.get("post", {}).get("no", 0))
        text = get_text(rec)
        if text:
            parts.append(f">>{pno}\n{text}")

    if not parts:
        return None

    context = "\n\n---\n\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
        {"role": "assistant", "content": reply_text},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/lgbt/posts.jsonl")
    parser.add_argument("--output", default="data/paired/lgbt/training_data.jsonl")
    parser.add_argument("--max-chain", type=int, default=MAX_CHAIN_DEPTH)
    parser.add_argument("--min-length", type=int, default=MIN_TEXT_LENGTH)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    posts: dict[int, dict] = {}
    ordered: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            post_no = int(record.get("post", {}).get("no", 0))
            posts[post_no] = record
            ordered.append(record)

    print(f"Loaded {len(posts)} posts")

    n_chain = 0
    n_multi = 0
    n_fallback = 0
    n_skip = 0

    with output_path.open("w", encoding="utf-8") as out:
        for record in ordered:
            if record.get("op"):
                continue

            post = record.get("post", {})
            post_no = int(post.get("no", 0))
            text = get_text(record)
            if not text or len(text) < args.min_length:
                n_skip += 1
                continue

            quotes = extract_quotes(text)
            valid_quotes = [q for q in quotes if q in posts and q != post_no]
            seen: set[int] = set()
            unique_quotes: list[int] = []
            for q in valid_quotes:
                if q not in seen:
                    seen.add(q)
                    unique_quotes.append(q)
            valid_quotes = unique_quotes

            if len(valid_quotes) <= 1:
                chain = build_chain(post_no, posts, args.max_chain)
                messages = chain_to_messages(chain)
                if messages:
                    out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                    if valid_quotes:
                        n_chain += 1
                    else:
                        n_fallback += 1
                else:
                    n_skip += 1
            else:
                quoted_records = [posts[q] for q in valid_quotes]
                messages = multi_quote_to_messages(record, quoted_records)
                if messages:
                    out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                    n_multi += 1
                else:
                    n_skip += 1

    total = n_chain + n_multi + n_fallback
    print(f"\nGenerated {total} training examples:")
    print(f"  Conversation chains (single-quote replies): {n_chain}")
    print(f"  Multi-quote replies: {n_multi}")
    print(f"  OP fallback (no explicit quote): {n_fallback}")
    print(f"  Skipped (too short / no parent): {n_skip}")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
