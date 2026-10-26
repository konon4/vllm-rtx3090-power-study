#!/usr/bin/env python3
"""Prefill ladder for an OpenAI-compatible vLLM server.

Sends prompts of an exact token length and records time to first token (TTFT).
With prefix caching defeated (every prompt is fresh random text) and a tiny
``max_tokens``, TTFT is a close proxy for the server-side prefill time.

Stages (same shape as the ladder in the README):

  S1   decode: 8 short prompts, 1024 output tokens each
  S2   bursts: 2 x 20K, 2 x 80K, 2 x 160K prompt tokens
  S3a  full:   132K, 207K, 248K prompt tokens, one at a time
  S3b  campaign: 8 x 131K back to back, no pause
  S3c  parallel: 3 x 131K sent at once

Stdlib only. Run GPU telemetry (scripts/gpu_telemetry.sh) next to it to get
power and temperature per stage.

Usage:
  python3 prefill_ladder.py --base-url http://localhost:8000 --out run.jsonl
  python3 prefill_ladder.py --stages S3a --out s3a.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import threading
import time
import urllib.request

STAGES = {
    "S1": {"kind": "decode", "n": 8, "prompt_words": 18, "max_tokens": 1024},
    "S2": {"kind": "prefill", "targets": [20_000, 20_000, 80_000, 80_000, 160_000, 160_000]},
    "S3a": {"kind": "prefill", "targets": [132_000, 207_000, 248_000]},
    "S3b": {"kind": "prefill", "targets": [131_000] * 8},
    "S3c": {"kind": "parallel", "targets": [131_000] * 3},
}

TOKENS_PER_WORD_GUESS = 3.35  # random pseudo-words tokenize to ~3.35 tokens each (measured)


def _post(url: str, payload: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def detect_model(base: str) -> str:
    return _get(f"{base}/v1/models")["data"][0]["id"]


def count_tokens(base: str, model: str, text: str) -> int:
    """vLLM exposes POST /tokenize; returns {"count": N, ...}."""
    return int(_post(f"{base}/tokenize", {"model": model, "prompt": text})["count"])


def random_words(rng: random.Random, n: int) -> str:
    letters = string.ascii_lowercase
    return " ".join("".join(rng.choice(letters) for _ in range(rng.randint(3, 9))) for _ in range(n))


def build_prompt(base: str, model: str, target: int, ceil: int, rng: random.Random,
                 tol: float = 0.01, max_iter: int = 6) -> tuple[str, int]:
    """Random text whose token count lands within +-tol of target and never above ceil."""
    words = max(1, int(target / TOKENS_PER_WORD_GUESS))
    text, n = "", 0
    for _ in range(max_iter):
        text = random_words(rng, words)
        n = count_tokens(base, model, text)
        if abs(n - target) <= tol * target and n <= ceil:
            return text, n
        words = max(1, int(words * target / max(n, 1)))
    if n > ceil:
        raise RuntimeError(f"could not build a prompt under ceil={ceil} (last {n} tokens)")
    return text, n


def stream_once(base: str, model: str, prompt: str, max_tokens: int, temperature: float = 0.0) -> dict:
    """One streaming completion. Returns TTFT, total time and output token count."""
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
               "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}
    req = urllib.request.Request(f"{base}/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if ttft is None and chunk.get("choices") and chunk["choices"][0].get("text"):
                ttft = time.perf_counter() - t0
            if chunk.get("usage"):
                usage = chunk["usage"]
    total = time.perf_counter() - t0
    return {"ttft_s": round(ttft if ttft is not None else total, 3), "total_s": round(total, 3),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


def run_stage(name: str, cfg: dict, base: str, model: str, ceil: int, rng: random.Random, out) -> None:
    def emit(rec: dict) -> None:
        rec.update(stage=name, wall_clock=time.strftime("%Y-%m-%dT%H:%M:%S"))
        if rec.get("prompt_tokens") and rec.get("ttft_s"):
            rec["prefill_tok_s"] = round(rec["prompt_tokens"] / rec["ttft_s"], 1)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(json.dumps(rec), file=sys.stderr)

    if cfg["kind"] == "decode":
        for _ in range(cfg["n"]):
            emit(stream_once(base, model, random_words(rng, cfg["prompt_words"]), cfg["max_tokens"]))
        return

    prompts = [build_prompt(base, model, t, ceil, rng) for t in cfg["targets"]]
    if cfg["kind"] == "prefill":
        for text, n in prompts:
            rec = stream_once(base, model, text, 16)
            rec["built_tokens"] = n
            emit(rec)
        return

    results: list[dict] = [{} for _ in prompts]

    def worker(i: int, text: str, n: int) -> None:
        results[i] = stream_once(base, model, text, 16)
        results[i]["built_tokens"] = n

    threads = [threading.Thread(target=worker, args=(i, t, n)) for i, (t, n) in enumerate(prompts)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for rec in results:
        rec["parallel"] = len(prompts)
        emit(rec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000", help="vLLM server root (no /v1)")
    ap.add_argument("--model", default=None, help="model id; default: first entry of /v1/models")
    ap.add_argument("--stages", default="S1,S2,S3a,S3b,S3c", help="comma-separated subset of " + ",".join(STAGES))
    ap.add_argument("--ceil", type=int, default=250_000, help="hard ceiling on prompt tokens (keep below max_model_len)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="JSONL output, one line per request")
    a = ap.parse_args()

    base = a.base_url.rstrip("/")
    model = a.model or detect_model(base)
    rng = random.Random(a.seed)
    wanted = [s.strip() for s in a.stages.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s): {unknown}")
    with open(a.out, "a", encoding="utf-8") as out:
        for name in wanted:
            print(f"== {name} start {time.strftime('%H:%M:%S')}", file=sys.stderr)
            run_stage(name, STAGES[name], base, model, a.ceil, rng, out)
            print(f"== {name} end {time.strftime('%H:%M:%S')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
