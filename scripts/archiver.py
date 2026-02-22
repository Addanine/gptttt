from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from email.utils import formatdate, parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

API_BASE = "https://a.4cdn.org"


def http_date(ts: float | int) -> str:
    return formatdate(int(ts), usegmt=True)


def parse_http_date(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).timestamp()
    except Exception:
        return None


def html_to_text(html: str | None) -> str:
    if not html:
        return ""
    text = html
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u00a0", " ")
    import html as html_mod

    return html_mod.unescape(text)


@dataclass
class ThreadState:
    last_modified_seen: int = 0
    last_fetched_modified: int = 0
    last_fetch_ts: float = 0.0
    last_post_no: int = 0


class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self.last_request = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()


class Archiver:
    def __init__(self, board: str, out_path: Path, state_path: Path, threads_interval: int,
                 min_thread_interval: int, archive_interval: int, user_agent: str,
                 rate_limit: float) -> None:
        self.board = board
        self.out_path = out_path
        self.state_path = state_path
        self.threads_interval = threads_interval
        self.min_thread_interval = min_thread_interval
        self.archive_interval = archive_interval
        self.user_agent = user_agent
        self.rate_limiter = RateLimiter(rate_limit)

        self.state: dict[str, Any] = {
            "board": board,
            "last_threads_mtime": 0,
            "last_threads_poll": 0.0,
            "last_archive_poll": 0.0,
            "threads": {},
        }
        self.last_state_save_ts = 0.0

    def load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            with self.state_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("board") != self.board:
                return
            self.state.update(data)
        except Exception:
            return

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False)
        tmp_path.replace(self.state_path)
        self.last_state_save_ts = time.time()

    def _touch_output_file(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.touch(exist_ok=True)

    def _save_state_periodically(self, every_seconds: float = 5.0) -> None:
        now = time.time()
        if now - self.last_state_save_ts >= every_seconds:
            self.save_state()

    def _fetch_json(self, client: httpx.Client, url: str, if_modified_since: int | None = None) -> tuple[Any | None, dict[str, str]]:
        headers = {"User-Agent": self.user_agent}
        if if_modified_since:
            headers["If-Modified-Since"] = http_date(if_modified_since)
        self.rate_limiter.wait()
        try:
            resp = client.get(url, headers=headers)
        except httpx.RequestError as exc:
            print(f"Request error: {exc}", file=sys.stderr)
            return None, {}
        if resp.status_code == 304:
            return None, dict(resp.headers)
        resp.raise_for_status()
        return resp.json(), dict(resp.headers)

    def _threads_url(self) -> str:
        return f"{API_BASE}/{self.board}/threads.json"

    def _thread_url(self, thread_id: int) -> str:
        return f"{API_BASE}/{self.board}/thread/{thread_id}.json"

    def _archive_url(self) -> str:
        return f"{API_BASE}/{self.board}/archive.json"

    def _ensure_thread_state(self, thread_id: int) -> ThreadState:
        threads = self.state.setdefault("threads", {})
        if str(thread_id) not in threads:
            threads[str(thread_id)] = ThreadState().__dict__
        t = threads[str(thread_id)]
        return ThreadState(
            last_modified_seen=int(t.get("last_modified_seen", 0)),
            last_fetched_modified=int(t.get("last_fetched_modified", 0)),
            last_fetch_ts=float(t.get("last_fetch_ts", 0.0)),
            last_post_no=int(t.get("last_post_no", 0)),
        )

    def _store_thread_state(self, thread_id: int, thread_state: ThreadState) -> None:
        threads = self.state.setdefault("threads", {})
        threads[str(thread_id)] = thread_state.__dict__

    def _poll_threads(self, client: httpx.Client) -> None:
        now = time.time()
        if now - float(self.state.get("last_threads_poll", 0.0)) < self.threads_interval:
            return

        last_mtime = int(self.state.get("last_threads_mtime", 0))
        data, headers = self._fetch_json(client, self._threads_url(), if_modified_since=last_mtime or None)
        self.state["last_threads_poll"] = now
        mtime = parse_http_date(headers.get("Last-Modified"))
        if mtime:
            self.state["last_threads_mtime"] = int(mtime)

        if data is None:
            return

        threads_before = len(self.state.get("threads", {}))
        for page in data:
            for thread in page.get("threads", []):
                thread_id = int(thread["no"])
                last_modified = int(thread.get("last_modified", 0))
                tstate = self._ensure_thread_state(thread_id)
                if last_modified > tstate.last_modified_seen:
                    tstate.last_modified_seen = last_modified
                    self._store_thread_state(thread_id, tstate)
        threads_after = len(self.state.get("threads", {}))
        if threads_before == 0 and threads_after > 0:
            print(f"Discovered {threads_after} active threads on /{self.board}/.")

    def _poll_archive(self, client: httpx.Client) -> list[int]:
        now = time.time()
        if now - float(self.state.get("last_archive_poll", 0.0)) < self.archive_interval:
            return []
        self.state["last_archive_poll"] = now
        data, _ = self._fetch_json(client, self._archive_url())
        if data is None:
            return []
        return [int(t) for t in data]

    def _append_posts(self, thread_id: int, posts: list[dict[str, Any]], last_post_no: int) -> int:
        if not posts:
            return last_post_no
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        new_last = last_post_no
        added = 0
        with self.out_path.open("a", encoding="utf-8") as f:
            for post in posts:
                no = int(post.get("no", 0))
                if no <= last_post_no:
                    continue
                record = {
                    "board": self.board,
                    "thread": thread_id,
                    "op": int(post.get("resto", 0)) == 0,
                    "post": post,
                    "com_text": html_to_text(post.get("com")),
                    "scraped_at": int(time.time()),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                added += 1
                if no > new_last:
                    new_last = no
        if added:
            print(f"Archived {added} post(s) from thread {thread_id}.")
        return new_last

    def _update_thread(self, client: httpx.Client, thread_id: int, tstate: ThreadState) -> ThreadState:
        if time.time() - tstate.last_fetch_ts < self.min_thread_interval:
            return tstate

        try:
            data, _ = self._fetch_json(
                client,
                self._thread_url(thread_id),
                if_modified_since=tstate.last_fetched_modified or None,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code in {404, 410}:
                tstate.last_fetched_modified = tstate.last_modified_seen
                return tstate
            raise
        tstate.last_fetch_ts = time.time()
        if data is None:
            tstate.last_fetched_modified = tstate.last_modified_seen
            return tstate

        posts = data.get("posts", [])
        if posts:
            thread_id = int(posts[0].get("no", thread_id))
        tstate.last_post_no = self._append_posts(thread_id, posts, tstate.last_post_no)
        tstate.last_fetched_modified = tstate.last_modified_seen
        return tstate

    def run(self) -> None:
        self.load_state()
        self._touch_output_file()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_state()

        with httpx.Client(timeout=20.0) as client:
            print(f"Archiving /{self.board}/ to {self.out_path}. Press Ctrl+C to stop.")
            try:
                while True:
                    self._poll_threads(client)
                    archived_threads = self._poll_archive(client)

                    now = time.time()
                    queue: list[int] = []
                    for thread_id_str, raw in list(self.state.get("threads", {}).items()):
                        thread_id = int(thread_id_str)
                        tstate = ThreadState(
                            last_modified_seen=int(raw.get("last_modified_seen", 0)),
                            last_fetched_modified=int(raw.get("last_fetched_modified", 0)),
                            last_fetch_ts=float(raw.get("last_fetch_ts", 0.0)),
                            last_post_no=int(raw.get("last_post_no", 0)),
                        )
                        if tstate.last_modified_seen > tstate.last_fetched_modified:
                            if now - tstate.last_fetch_ts >= self.min_thread_interval:
                                queue.append(thread_id)

                        if thread_id in archived_threads and tstate.last_modified_seen == 0:
                            queue.append(thread_id)

                    if queue:
                        thread_id = queue[0]
                        tstate = self._ensure_thread_state(thread_id)
                        updated = self._update_thread(client, thread_id, tstate)
                        self._store_thread_state(thread_id, updated)
                        self.save_state()
                        continue

                    self._save_state_periodically()
                    time.sleep(0.5)

            except KeyboardInterrupt:
                print("\nStopping. Saving state...")
                self.save_state()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default="lgbt")
    parser.add_argument("--out", default="data/raw/{board}/posts.jsonl")
    parser.add_argument("--state", default="data/raw/{board}/state.json")
    parser.add_argument("--threads-interval", type=int, default=30)
    parser.add_argument("--archive-interval", type=int, default=300)
    parser.add_argument("--min-thread-interval", type=int, default=15)
    parser.add_argument("--rate-limit", type=float, default=1.05)
    parser.add_argument("--user-agent", default="chan-archive/0.1 (contact: local)")

    args = parser.parse_args()
    out_path = Path(args.out.format(board=args.board))
    state_path = Path(args.state.format(board=args.board))

    archiver = Archiver(
        board=args.board,
        out_path=out_path,
        state_path=state_path,
        threads_interval=args.threads_interval,
        min_thread_interval=args.min_thread_interval,
        archive_interval=args.archive_interval,
        user_agent=args.user_agent,
        rate_limit=args.rate_limit,
    )
    archiver.run()


if __name__ == "__main__":
    main()
