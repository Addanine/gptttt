from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
LORA_RANK = 32

SYSTEM_PROMPT = (
    "You are an anonymous poster on /lgbt/. "
    "Reply to the post naturally, matching the tone and style of the board."
)

BOLD = "\033[1m"
DIM = "\033[90m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
SEPARATOR = "─" * 72
HEAVY_SEP = "━" * 72

BENCHMARK_SUITE: dict[str, list[str]] = {
    "greentext": [
        ">be me\n>mtf, 6 months hrt\n>voice training going nowhere\n>sound like a dude on the phone every time\nis it over?",
        ">be me, ftm\n>finally got top surgery\n>wake up from anaesthesia\n>first thing i do is look down\n>flat\n>start crying tears of joy\n>nurse thinks something went wrong",
        ">be me\n>closeted at work\n>coworker starts talking about how they hate trans people\n>mfw i have to smile and nod\nwhat do",
    ],
    "advice_seeking": [
        "how do i know if im actually trans or just a confused femboy",
        "i want to start hrt but im scared of losing my family. what would you do?",
        "my therapist says i might have gender dysphoria but im not sure. how did you all figure it out?",
        "is it worth transitioning at 30? i feel like its too late",
    ],
    "venting": [
        "i came out to my parents today and they said they still love me but i can tell theyre disappointed",
        "3 years hrt and i still dont pass. i try so hard every single day and nothing changes. im so tired",
        "just got misgendered at the store again. the cashier called me sir right to my face. i wanted to disappear",
        "my best friend ghosted me after i came out. 10 years of friendship gone just like that",
    ],
    "discussion": [
        "do you think passing privilege is real? like are trans people who pass treated fundamentally differently",
        "hot take: the trans community online is way too focused on aesthetics and not enough on actual wellbeing",
        "why does everyone here act like being trans is the worst thing ever. i actually like being trans",
        "thoughts on self-medicating hrt? i know its not ideal but some people dont have access to doctors",
    ],
    "shitposting": [
        "just mass mass mass",
        "hon detected",
        "daily reminder that if you have to ask if you pass, you dont",
        "imagine not being on injections in 2026 lmao",
    ],
    "short_reply": [
        "is it over?",
        "am i valid",
        "cope",
        "ywnbaw",
        "thoughts?",
    ],
    "controversial": [
        "do non-binary people belong in trans spaces?",
        "truscum vs tucute debate is so tired. can we move on",
        "should trans women compete in womens sports",
        "is being trans a choice",
    ],
}


@dataclass
class GenerationResult:
    prompt: str
    replies: list[str]
    temperature: float
    max_tokens: int
    num_samples: int
    elapsed_s: float
    mode: str
    category: str = ""
    metadata: dict = field(default_factory=dict)


class ModelTester:
    def __init__(self, model_path: str, base_model: str = MODEL_NAME, rank: int = LORA_RANK):
        self.model_path = model_path
        self.base_model = base_model

        print(f"  Loading tokenizer for {base_model}...")
        self.tokenizer = get_tokenizer(base_model)
        renderer_name = model_info.get_recommended_renderer_name(base_model)
        self.renderer = renderers.get_renderer(renderer_name, self.tokenizer)
        print(f"  Renderer: {renderer_name}")

        print("  Connecting to Tinker...")
        service_client = tinker.ServiceClient()
        training_client = service_client.create_lora_training_client(
            base_model=base_model,
            rank=rank,
        )
        self.sampling_client = training_client.create_sampling_client(model_path=model_path)
        print(f"  Model loaded: {model_path}")

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 200,
        num_samples: int = 1,
        top_p: float = 0.95,
    ) -> tuple[list[str], float]:
        prompt = self.renderer.build_generation_prompt(messages)
        stop_seqs = self.renderer.get_stop_sequences()

        t0 = time.time()
        result = self.sampling_client.sample(
            prompt=prompt,
            sampling_params=types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_seqs,
                top_p=top_p,
            ),
            num_samples=num_samples,
        ).result()
        elapsed = time.time() - t0

        replies = []
        for seq in result.sequences:
            text = self.tokenizer.decode(seq.tokens, skip_special_tokens=True).strip()
            replies.append(text)

        return replies, elapsed

    def reply_to_post(
        self,
        post: str,
        history: list[dict] | None = None,
        **kwargs,
    ) -> tuple[list[str], float]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": post})
        return self.generate(messages, **kwargs)


@dataclass
class Settings:
    temperature: float = 0.7
    max_tokens: int = 200
    num_samples: int = 3
    top_p: float = 0.95


def clear_screen():
    print("\033[2J\033[H", end="")


def fmt_post(post: str) -> str:
    lines = post.split("\n")
    formatted = []
    for line in lines:
        if line.startswith(">"):
            formatted.append(f"    {GREEN}{line}{RESET}")
        else:
            formatted.append(f"    {line}")
    return "\n".join(formatted)


def fmt_reply(reply: str, idx: int | None = None) -> str:
    prefix = f"    {CYAN}Reply {idx}:{RESET} " if idx is not None else f"    {CYAN}>{RESET} "
    lines = reply.split("\n")
    result = [f"{prefix}{lines[0]}"]
    for line in lines[1:]:
        result.append(f"      {line}")
    return "\n".join(result)


def print_generation(post: str, replies: list[str], elapsed: float, category: str = ""):
    label = f" {DIM}[{category}]{RESET}" if category else ""
    print(f"\n  {BOLD}Post{label}:{RESET}")
    print(fmt_post(post))
    print(f"    {DIM}({elapsed:.2f}s, {len(replies)} sample{'s' if len(replies) != 1 else ''}){RESET}")
    print(f"  {SEPARATOR}")
    for i, reply in enumerate(replies, 1):
        print(fmt_reply(reply, i if len(replies) > 1 else None))
        if i < len(replies):
            print()
    print()


def show_banner():
    print(f"""
{HEAVY_SEP}
{BOLD}  /lgbt/ Model Tester{RESET}
{HEAVY_SEP}
""")


def show_menu(title: str, options: list[str], back: bool = True) -> str:
    print(f"\n  {BOLD}{title}{RESET}")
    print(f"  {SEPARATOR}")
    for i, opt in enumerate(options, 1):
        print(f"    {YELLOW}{i}{RESET}) {opt}")
    if back:
        print(f"    {DIM}0) Back{RESET}")
    print()

    while True:
        try:
            raw = input(f"  {BOLD}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            return "back"
        if raw == "0" and back:
            return "back"
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"    {RED}Pick 1-{len(options)}{' or 0' if back else ''}{RESET}")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" {DIM}[{default}]{RESET}" if default else ""
    try:
        val = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return val or default


def ask_int(prompt: str, default: int) -> int:
    raw = ask(prompt, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def ask_float(prompt: str, default: float) -> float:
    raw = ask(prompt, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def pause():
    try:
        input(f"\n  {DIM}Press Enter to continue...{RESET}")
    except (EOFError, KeyboardInterrupt):
        pass


def do_interactive(tester: ModelTester, s: Settings, results: list[GenerationResult]):
    history: list[dict] = []
    last_post: str | None = None

    print(f"\n{HEAVY_SEP}")
    print(f"  {BOLD}Interactive Chat{RESET}")
    print(f"{HEAVY_SEP}")
    print(f"  Type a post and get a reply. Conversation is remembered.")
    print(f"  Commands: {DIM}/reset /history /redo /samples N /temp F /tokens N /quit{RESET}\n")

    while True:
        try:
            user_input = input(f"  {BOLD}[You]{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split()
            match cmd[0]:
                case "/quit" | "/exit" | "/q":
                    break
                case "/reset":
                    history.clear()
                    print(f"    {DIM}History cleared.{RESET}\n")
                    continue
                case "/history":
                    if not history:
                        print(f"    {DIM}(empty){RESET}\n")
                    else:
                        for msg in history:
                            r = msg["role"]
                            tag = CYAN if r == "assistant" else YELLOW
                            print(f"    {tag}[{r}]{RESET} {msg['content'][:120]}...")
                        print()
                    continue
                case "/samples":
                    if len(cmd) > 1:
                        s.num_samples = max(1, int(cmd[1]))
                    print(f"    {DIM}Samples: {s.num_samples}{RESET}\n")
                    continue
                case "/temp":
                    if len(cmd) > 1:
                        s.temperature = max(0.0, min(2.0, float(cmd[1])))
                    print(f"    {DIM}Temperature: {s.temperature}{RESET}\n")
                    continue
                case "/tokens":
                    if len(cmd) > 1:
                        s.max_tokens = max(1, int(cmd[1]))
                    print(f"    {DIM}Max tokens: {s.max_tokens}{RESET}\n")
                    continue
                case "/redo":
                    if last_post is None:
                        print(f"    {DIM}Nothing to redo.{RESET}\n")
                        continue
                    if history and history[-1]["role"] == "assistant":
                        history.pop()
                    user_input = last_post
                case _:
                    print(f"    {RED}Unknown command: {cmd[0]}{RESET}\n")
                    continue

        last_post = user_input
        replies, elapsed = tester.reply_to_post(
            user_input,
            history=history,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            num_samples=s.num_samples,
        )
        print_generation(user_input, replies, elapsed)

        history.append({"role": "user", "content": user_input})
        if replies:
            history.append({"role": "assistant", "content": replies[0]})

        results.append(GenerationResult(
            prompt=user_input, replies=replies, temperature=s.temperature,
            max_tokens=s.max_tokens, num_samples=s.num_samples, elapsed_s=elapsed,
            mode="interactive",
        ))


def do_single(tester: ModelTester, s: Settings, results: list[GenerationResult]):
    print(f"\n  {BOLD}Single Post{RESET}")
    print(f"  {SEPARATOR}")
    prompt = ask("Enter your post")
    if not prompt:
        return

    replies, elapsed = tester.reply_to_post(
        prompt,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        num_samples=s.num_samples,
    )
    print_generation(prompt, replies, elapsed)
    results.append(GenerationResult(
        prompt=prompt, replies=replies, temperature=s.temperature,
        max_tokens=s.max_tokens, num_samples=s.num_samples, elapsed_s=elapsed,
        mode="single",
    ))
    pause()


def do_batch(tester: ModelTester, s: Settings, results: list[GenerationResult]):
    print(f"\n  {BOLD}Batch Mode{RESET}")
    print(f"  {SEPARATOR}")
    filepath = ask("Path to prompts file")
    if not filepath:
        return
    path = Path(filepath)
    if not path.exists():
        print(f"    {RED}File not found: {path}{RESET}")
        pause()
        return

    prompts: list[str] = []
    current_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "" and current_lines:
            prompts.append("\n".join(current_lines))
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        prompts.append("\n".join(current_lines))

    print(f"    Loaded {BOLD}{len(prompts)}{RESET} prompts\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"  {DIM}[{i}/{len(prompts)}]{RESET}")
        replies, elapsed = tester.reply_to_post(
            prompt,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            num_samples=s.num_samples,
        )
        print_generation(prompt, replies, elapsed)
        results.append(GenerationResult(
            prompt=prompt, replies=replies, temperature=s.temperature,
            max_tokens=s.max_tokens, num_samples=s.num_samples, elapsed_s=elapsed,
            mode="batch",
        ))
    pause()


def do_benchmark(tester: ModelTester, s: Settings, results: list[GenerationResult]):
    cat_names = list(BENCHMARK_SUITE.keys())
    print(f"\n  {BOLD}Benchmark Suite{RESET}")
    print(f"  {SEPARATOR}")
    print(f"    {YELLOW}a{RESET}) Run ALL categories")
    for i, name in enumerate(cat_names, 1):
        count = len(BENCHMARK_SUITE[name])
        print(f"    {YELLOW}{i}{RESET}) {name} ({count} prompts)")
    print(f"    {DIM}0) Back{RESET}\n")

    raw = ask("Pick categories (comma-separated, or 'a' for all)")
    if not raw or raw == "0":
        return

    if raw.lower() == "a":
        categories = cat_names
    else:
        categories = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= len(cat_names):
                categories.append(cat_names[int(part) - 1])
            elif part in cat_names:
                categories.append(part)

    if not categories:
        print(f"    {RED}No valid categories selected.{RESET}")
        return

    total_prompts = sum(len(BENCHMARK_SUITE[c]) for c in categories)
    print(f"\n    Running {BOLD}{len(categories)}{RESET} categories, {BOLD}{total_prompts}{RESET} prompts total\n")

    bench_results: list[GenerationResult] = []
    idx = 0

    for category in categories:
        prompts = BENCHMARK_SUITE[category]
        print(f"  {BOLD}{'=' * 56}{RESET}")
        print(f"  {BOLD}  {category} ({len(prompts)} prompts){RESET}")
        print(f"  {BOLD}{'=' * 56}{RESET}")

        for prompt in prompts:
            idx += 1
            print(f"  {DIM}[{idx}/{total_prompts}]{RESET}")
            replies, elapsed = tester.reply_to_post(
                prompt,
                temperature=s.temperature,
                max_tokens=s.max_tokens,
                num_samples=s.num_samples,
            )
            print_generation(prompt, replies, elapsed, category=category)
            r = GenerationResult(
                prompt=prompt, replies=replies, temperature=s.temperature,
                max_tokens=s.max_tokens, num_samples=s.num_samples, elapsed_s=elapsed,
                mode="benchmark", category=category,
            )
            bench_results.append(r)
            results.append(r)

    print(f"\n{HEAVY_SEP}")
    print(f"  {BOLD}Benchmark Summary{RESET}")
    print(f"{HEAVY_SEP}")
    by_cat: dict[str, list[GenerationResult]] = {}
    for r in bench_results:
        by_cat.setdefault(r.category, []).append(r)
    total_time = sum(r.elapsed_s for r in bench_results)
    total_replies = sum(len(r.replies) for r in bench_results)
    avg_len = (
        sum(len(reply) for r in bench_results for reply in r.replies) / total_replies
        if total_replies > 0 else 0
    )

    for cat, cat_r in by_cat.items():
        ct = sum(r.elapsed_s for r in cat_r)
        cr = sum(len(r.replies) for r in cat_r)
        cl = sum(len(reply) for r in cat_r for reply in r.replies) / cr if cr > 0 else 0
        print(f"    {cat:20s}  {len(cat_r):3d} prompts  {ct:6.1f}s  avg len: {cl:.0f}")

    print(f"  {SEPARATOR}")
    print(f"    {'TOTAL':20s}  {len(bench_results):3d} prompts  {total_time:6.1f}s  avg len: {avg_len:.0f}")
    print()
    pause()


def do_thread(tester: ModelTester, s: Settings, results: list[GenerationResult]):
    print(f"\n  {BOLD}Thread Simulation{RESET}")
    print(f"  {SEPARATOR}")
    op = ask("Opening post (OP)", "new thread. whats on your mind today /lgbt/?")
    rounds = ask_int("How many replies to generate", 5)

    print(f"\n  {BOLD}OP:{RESET}")
    print(fmt_post(op))
    print(f"  {SEPARATOR}")

    history: list[dict] = []
    last_text = op

    for i in range(rounds):
        replies, elapsed = tester.reply_to_post(
            last_text,
            history=history,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            num_samples=1,
        )
        reply = replies[0] if replies else "(no output)"

        print(f"\n  {BOLD}Anon #{i + 1}:{RESET} {DIM}({elapsed:.2f}s){RESET}")
        print(fmt_reply(reply))

        history.append({"role": "user", "content": last_text})
        history.append({"role": "assistant", "content": reply})
        last_text = reply

        results.append(GenerationResult(
            prompt=last_text, replies=replies, temperature=s.temperature,
            max_tokens=s.max_tokens, num_samples=1, elapsed_s=elapsed,
            mode="thread", metadata={"round": i + 1, "op": op},
        ))

    print(f"\n  {DIM}Thread complete: {rounds} replies.{RESET}")
    pause()


def do_sweep(tester: ModelTester, s: Settings, results: list[GenerationResult]):
    print(f"\n  {BOLD}Temperature Sweep{RESET}")
    print(f"  {SEPARATOR}")
    prompt = ask("Prompt", "do you think hrt changes your personality or just your body?")
    temps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]

    print(f"\n    Prompt: {BOLD}{prompt}{RESET}")
    print(f"    Temps:  {temps}")
    print(f"    Samples per temp: {s.num_samples}\n")

    sweep_results: list[GenerationResult] = []

    for temp in temps:
        replies, elapsed = tester.reply_to_post(
            prompt,
            temperature=temp,
            max_tokens=s.max_tokens,
            num_samples=s.num_samples,
        )
        print(f"  {BOLD}T={temp:.1f}{RESET} {DIM}({elapsed:.2f}s){RESET}")
        for i, reply in enumerate(replies, 1):
            label = f"    {DIM}[{i}]{RESET} " if len(replies) > 1 else "    "
            print(f"{label}{reply}")
        print()

        r = GenerationResult(
            prompt=prompt, replies=replies, temperature=temp,
            max_tokens=s.max_tokens, num_samples=s.num_samples, elapsed_s=elapsed,
            mode="sweep", metadata={"temperature": temp},
        )
        sweep_results.append(r)
        results.append(r)

    print(f"  {BOLD}Diversity Analysis:{RESET}")
    for r in sweep_results:
        t = r.metadata["temperature"]
        lengths = [len(reply) for reply in r.replies]
        avg = sum(lengths) / len(lengths) if lengths else 0
        uniq = len(set(reply[:30] for reply in r.replies))
        print(f"    T={t:.1f}  avg_len={avg:5.0f}  unique_starts={uniq}/{len(r.replies)}")
    print()
    pause()


def do_versus(tester: ModelTester, s: Settings, results: list[GenerationResult]):
    print(f"\n  {BOLD}Versus Mode{RESET}")
    print(f"  {SEPARATOR}")
    prompt = ask("Prompt", "i think i might be trans but im not sure")
    n = ask_int("How many samples", 8)

    replies, elapsed = tester.reply_to_post(
        prompt,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        num_samples=n,
    )

    print(f"\n    Prompt: {BOLD}{prompt}{RESET}")
    print(f"    {DIM}{len(replies)} samples in {elapsed:.2f}s{RESET}")
    print(f"  {SEPARATOR}")

    for i, reply in enumerate(replies, 1):
        print(f"\n  {BOLD}Sample {i}:{RESET}")
        for line in reply.split("\n"):
            print(f"      {line}")

    print(f"\n  {SEPARATOR}")
    lengths = [len(r) for r in replies]
    uniq = len(set(r[:50] for r in replies))
    print(f"    Length: {min(lengths)}-{max(lengths)} chars (avg {sum(lengths)/len(lengths):.0f})")
    print(f"    Unique openings: {uniq}/{len(replies)}")
    print()

    results.append(GenerationResult(
        prompt=prompt, replies=replies, temperature=s.temperature,
        max_tokens=s.max_tokens, num_samples=n, elapsed_s=elapsed,
        mode="versus",
    ))
    pause()


def do_stress(tester: ModelTester, s: Settings, results: list[GenerationResult]):
    print(f"\n  {BOLD}Stress Test{RESET}")
    print(f"  {SEPARATOR}")
    count = ask_int("Number of generations", 50)

    prompts_pool = [p for ps in BENCHMARK_SUITE.values() for p in ps]
    random.seed(42)
    random.shuffle(prompts_pool)

    stress_results: list[GenerationResult] = []
    total_words = 0
    t_start = time.time()

    for i in range(count):
        prompt = prompts_pool[i % len(prompts_pool)]
        replies, elapsed = tester.reply_to_post(
            prompt,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            num_samples=1,
        )
        reply = replies[0] if replies else ""
        total_words += len(reply.split())

        preview = reply.replace("\n", " ")[:80]
        status = f"{GREEN}OK{RESET}" if len(reply) > 10 else f"{RED}SHORT{RESET}"
        print(f"    [{i+1:3d}/{count}] {elapsed:5.2f}s  {status}  {DIM}{preview}{RESET}")

        r = GenerationResult(
            prompt=prompt, replies=replies, temperature=s.temperature,
            max_tokens=s.max_tokens, num_samples=1, elapsed_s=elapsed,
            mode="stress",
        )
        stress_results.append(r)
        results.append(r)

    wall = time.time() - t_start

    print(f"\n{HEAVY_SEP}")
    print(f"  {BOLD}Stress Test Results{RESET}")
    print(f"{HEAVY_SEP}")
    times = [r.elapsed_s for r in stress_results]
    lens = [len(r.replies[0]) for r in stress_results if r.replies]
    empty = sum(1 for r in stress_results if not r.replies or len(r.replies[0]) < 5)
    openings = [r.replies[0][:40] for r in stress_results if r.replies]
    uniq_o = len(set(openings))

    print(f"    Total:       {count} gens in {wall:.1f}s")
    print(f"    Throughput:  {count / wall:.2f} gen/s")
    print(f"    Latency:     min={min(times):.2f}s  avg={sum(times)/len(times):.2f}s  max={max(times):.2f}s")
    if lens:
        print(f"    Reply len:   min={min(lens)}  avg={sum(lens)/len(lens):.0f}  max={max(lens)}")
    print(f"    Empty/short: {empty}/{count}")
    print(f"    ~Words:      {total_words}")
    print(f"    Unique opens: {uniq_o}/{len(openings)} ({uniq_o/len(openings)*100:.0f}%)")
    print()
    pause()


def do_random_from_data(tester: ModelTester, s: Settings, results: list[GenerationResult]):
    data_path = Path("data/paired/lgbt/training_data.jsonl")
    if not data_path.exists():
        print(f"    {RED}Training data not found at {data_path}{RESET}")
        pause()
        return

    print(f"\n  {BOLD}Random Real Post{RESET}")
    print(f"  {SEPARATOR}")
    count = ask_int("How many random posts to test", 5)

    all_user_posts: list[str] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            msgs = entry.get("messages", [])
            for m in msgs:
                if m["role"] == "user":
                    all_user_posts.append(m["content"])
                    break

    if not all_user_posts:
        print(f"    {RED}No user posts found in data.{RESET}")
        pause()
        return

    sampled = random.sample(all_user_posts, min(count, len(all_user_posts)))
    print(f"    Picked {len(sampled)} random posts from {len(all_user_posts)} total\n")

    for i, post in enumerate(sampled, 1):
        display = post if len(post) < 500 else post[:500] + "..."
        print(f"  {DIM}[{i}/{len(sampled)}]{RESET}")
        replies, elapsed = tester.reply_to_post(
            post,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            num_samples=s.num_samples,
        )
        print_generation(display, replies, elapsed)
        results.append(GenerationResult(
            prompt=post, replies=replies, temperature=s.temperature,
            max_tokens=s.max_tokens, num_samples=s.num_samples, elapsed_s=elapsed,
            mode="random_real",
        ))
    pause()


def do_settings(s: Settings):
    while True:
        print(f"\n  {BOLD}Settings{RESET}")
        print(f"  {SEPARATOR}")
        print(f"    {YELLOW}1{RESET}) Temperature     {BOLD}{s.temperature}{RESET}")
        print(f"    {YELLOW}2{RESET}) Max tokens       {BOLD}{s.max_tokens}{RESET}")
        print(f"    {YELLOW}3{RESET}) Samples/prompt   {BOLD}{s.num_samples}{RESET}")
        print(f"    {YELLOW}4{RESET}) Top-p            {BOLD}{s.top_p}{RESET}")
        print(f"    {DIM}0) Back{RESET}\n")

        raw = ask(">")
        match raw:
            case "0" | "":
                return
            case "1":
                s.temperature = ask_float("Temperature (0.0 - 2.0)", s.temperature)
                s.temperature = max(0.0, min(2.0, s.temperature))
            case "2":
                s.max_tokens = ask_int("Max tokens", s.max_tokens)
                s.max_tokens = max(1, s.max_tokens)
            case "3":
                s.num_samples = ask_int("Samples per prompt", s.num_samples)
                s.num_samples = max(1, s.num_samples)
            case "4":
                s.top_p = ask_float("Top-p (0.0 - 1.0)", s.top_p)
                s.top_p = max(0.0, min(1.0, s.top_p))
            case _:
                print(f"    {RED}Pick 1-4 or 0{RESET}")


def export_results(results: list[GenerationResult]):
    if not results:
        print(f"    {DIM}No results to export.{RESET}")
        pause()
        return
    path = ask("Export path", "results.json")
    out = Path(path)
    data = [asdict(r) for r in results]
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    {GREEN}Saved {len(results)} results to {out}{RESET}")
    pause()


MAIN_MENU_OPTIONS = [
    "Interactive chat",
    "Single post",
    "Batch (from file)",
    "Benchmark suite",
    "Thread simulation",
    "Temperature sweep",
    "Versus (multi-sample)",
    "Stress test",
    "Random real post from data",
    "Settings",
    "Export results",
    "Quit",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model", default=MODEL_NAME)
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    args = parser.parse_args()

    show_banner()

    tester = ModelTester(
        model_path=args.model_path,
        base_model=args.base_model,
        rank=args.rank,
    )

    settings = Settings()
    all_results: list[GenerationResult] = []

    while True:
        print(f"\n  {BOLD}Main Menu{RESET}  {DIM}(temp={settings.temperature} samples={settings.num_samples} tokens={settings.max_tokens}){RESET}")
        print(f"  {SEPARATOR}")
        for i, opt in enumerate(MAIN_MENU_OPTIONS, 1):
            color = RED if opt == "Quit" else YELLOW
            print(f"    {color}{i:2d}{RESET}) {opt}")
        print()

        try:
            raw = input(f"  {BOLD}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw.isdigit() or not (1 <= int(raw) <= len(MAIN_MENU_OPTIONS)):
            print(f"    {RED}Pick 1-{len(MAIN_MENU_OPTIONS)}{RESET}")
            continue

        choice = int(raw)

        match choice:
            case 1:
                do_interactive(tester, settings, all_results)
            case 2:
                do_single(tester, settings, all_results)
            case 3:
                do_batch(tester, settings, all_results)
            case 4:
                do_benchmark(tester, settings, all_results)
            case 5:
                do_thread(tester, settings, all_results)
            case 6:
                do_sweep(tester, settings, all_results)
            case 7:
                do_versus(tester, settings, all_results)
            case 8:
                do_stress(tester, settings, all_results)
            case 9:
                do_random_from_data(tester, settings, all_results)
            case 10:
                do_settings(settings)
            case 11:
                export_results(all_results)
            case 12:
                if all_results:
                    save = ask("Export results before quitting? (y/n)", "n")
                    if save.lower() == "y":
                        export_results(all_results)
                print("  Bye!")
                break

    print()


if __name__ == "__main__":
    main()
