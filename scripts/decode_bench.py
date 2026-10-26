#!/usr/bin/env python3
"""Decode-speed probe for an OpenAI-compatible server: output tokens per second.

Used to compare one server launched with and without speculative decoding
(e.g. vLLM ``--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'``
against no speculative config), at several sampling temperatures.

Decode speed = completion tokens after the first chunk / (time of last chunk - time of
first chunk). This excludes prefill, so it is comparable across prompt lengths.

Why several temperatures: draft acceptance falls as temperature rises, so a speedup
measured only at temperature 0 overstates what an agent running at 0.6-1.0 will see.

Usage:
  python3 decode_bench.py --base-url http://localhost:8000 --temps 0,0.6,1.0 --runs 2 --out mtp.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import string
import sys
import time
import urllib.request


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def one(base: str, model: str, prompt: str, max_tokens: int, temperature: float) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
               "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{base}/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_last = None
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
            if chunk.get("choices") and chunk["choices"][0].get("text"):
                now = time.perf_counter()
                t_first = t_first or now
                t_last = now
            if chunk.get("usage"):
                usage = chunk["usage"]
    n_out = int(usage.get("completion_tokens") or 0)
    span = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    return {"temperature": temperature, "ttft_s": round((t_first or time.perf_counter()) - t0, 3),
            "completion_tokens": n_out, "prompt_tokens": usage.get("prompt_tokens"),
            "decode_tok_s": round((n_out - 1) / span, 1) if span and n_out > 1 else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temps", default="0,0.6,1.0")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--prompt-words", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--label", default="", help="free text stored with each row, e.g. 'mtp3' or 'no-spec'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = a.base_url.rstrip("/")
    model = a.model or _get(f"{base}/v1/models")["data"][0]["id"]
    rng = random.Random(0)
    temps = [float(t) for t in a.temps.split(",")]
    with open(a.out, "a", encoding="utf-8") as out:
        for t in temps:
            speeds = []
            for i in range(a.warmup + a.runs):
                # Fresh prefix per request so the prefix cache cannot help.
                prefix = "".join(rng.choice(string.ascii_lowercase) for _ in range(12))
                prompt = prefix + " Write a long, detailed technical essay about distributed systems. " \
                    + " ".join("context" for _ in range(a.prompt_words))
                rec = one(base, model, prompt, a.max_tokens, t)
                rec.update(label=a.label, warmup=i < a.warmup)
                out.write(json.dumps(rec) + "\n")
                if i >= a.warmup and rec["decode_tok_s"]:
                    speeds.append(rec["decode_tok_s"])
            med = statistics.median(speeds) if speeds else None
            print(f"label={a.label or '-'} temperature={t}: decode tok/s per run {speeds}, median {med}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
