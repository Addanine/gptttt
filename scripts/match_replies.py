from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable


QUOTE_RE = re.compile(r">>(\d+)")


def html_to_text(html: str | None) -> str:
    if not html:
        return ""
    text = html
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"<[^>]+>", "", text)
    import html as html_mod

    return html_mod.unescape(text)


def post_text(record: dict[str, Any]) -> str:
    post = record.get("post", {})
    sub = post.get("sub") or ""
    com_text = record.get("com_text") or html_to_text(post.get("com"))
    parts = [p for p in [sub.strip(), com_text.strip()] if p]
    return "\n\n".join(parts).strip()


def ensure_index(db_path: Path, jsonl_path: Path) -> None:
    jsonl_mtime = int(jsonl_path.stat().st_mtime)

    if db_path.exists():
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
            if meta.get("input_path") == str(jsonl_path) and int(meta.get("input_mtime", "0")) == jsonl_mtime:
                return
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            """
            CREATE TABLE posts (
              no INTEGER PRIMARY KEY,
              board TEXT,
              thread INTEGER,
              resto INTEGER,
              time INTEGER,
              name TEXT,
              sub TEXT,
              com_html TEXT,
              com_text TEXT,
              text TEXT
            )
            """
        )
        db.execute("CREATE INDEX idx_posts_thread ON posts(thread)")
        db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")

        batch: list[tuple[Any, ...]] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                post = record.get("post", {})
                text = post_text(record)
                batch.append(
                    (
                        int(post.get("no", 0)),
                        record.get("board"),
                        int(record.get("thread", 0)),
                        int(post.get("resto", 0)),
                        int(post.get("time", 0)),
                        post.get("name"),
                        post.get("sub"),
                        post.get("com"),
                        record.get("com_text"),
                        text,
                    )
                )
                if len(batch) >= 1000:
                    db.executemany(
                        """
                        INSERT OR REPLACE INTO posts
                        (no, board, thread, resto, time, name, sub, com_html, com_text, text)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    batch.clear()
        if batch:
            db.executemany(
                """
                INSERT OR REPLACE INTO posts
                (no, board, thread, resto, time, name, sub, com_html, com_text, text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )

        db.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            [
                ("input_path", str(jsonl_path)),
                ("input_mtime", str(jsonl_mtime)),
                ("created_at", str(int(time.time()))),
            ],
        )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def fetch_post_text(db: sqlite3.Connection, post_no: int) -> tuple[str | None, dict[str, Any] | None]:
    row = db.execute(
        "SELECT no, board, thread, resto, time, name, sub, text FROM posts WHERE no = ?",
        (post_no,),
    ).fetchone()
    if not row:
        return None, None
    no, board, thread, resto, ts, name, sub, text = row
    meta = {
        "no": no,
        "board": board,
        "thread": thread,
        "resto": resto,
        "time": ts,
        "name": name,
        "sub": sub,
    }
    return text, meta


def extract_quote_nos(text: str) -> list[int]:
    return [int(m.group(1)) for m in QUOTE_RE.finditer(text)]


def emit_pairs(jsonl_path: Path, db_path: Path, out_path: Path, include_op_fallback: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as db, out_path.open("w", encoding="utf-8") as out:
        for record in iter_jsonl(jsonl_path):
            post = record.get("post", {})
            resto = int(post.get("resto", 0))
            if resto == 0:
                continue

            child_no = int(post.get("no", 0))
            child_text = post_text(record)
            if not child_text:
                continue

            quotes = extract_quote_nos(child_text)
            parents = quotes[:]
            if not parents and include_op_fallback:
                parents = [resto]

            seen_parents: set[int] = set()
            for parent_no in parents:
                if parent_no in seen_parents:
                    continue
                seen_parents.add(parent_no)
                if parent_no == child_no:
                    continue
                parent_text, parent_meta = fetch_post_text(db, parent_no)
                if not parent_text:
                    continue

                edge = "quote" if parent_no in quotes else "resto"
                out.write(
                    json.dumps(
                        {
                            "board": record.get("board"),
                            "thread": record.get("thread"),
                            "edge": edge,
                            "parent_no": parent_no,
                            "child_no": child_no,
                            "messages": [
                                {"role": "user", "content": parent_text},
                                {"role": "assistant", "content": child_text},
                            ],
                            "parent_meta": parent_meta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/lgbt/posts.jsonl")
    parser.add_argument("--output", default="data/paired/lgbt/pairs.jsonl")
    parser.add_argument("--index", default="data/index/lgbt_posts.sqlite3")
    parser.add_argument("--include-op-fallback", action="store_true")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    index_path = Path(args.index)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    ensure_index(index_path, input_path)
    emit_pairs(input_path, index_path, output_path, args.include_op_fallback)


if __name__ == "__main__":
    main()
