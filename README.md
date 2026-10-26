# vLLM on 2x RTX 3090: power cap vs prefill and decode

Measurements of how the per-card power limit changes prefill and decode speed for a 35B
MoE model served by vLLM with tensor parallelism (TP=2) on two consumer RTX 3090 cards,
plus a sustained-load test of the same machine and a speculative-decoding (MTP) run.

Short version:

- Prefill is compute-bound and follows the power cap: 3.3K tok/s at 200 W per card,
  4.2K tok/s at 250 W on a 132K-token prompt (+27 %). Going to 300 W adds only 4-5 %
  at 132K and about 10 % at ~200K.
- Decode is memory-bandwidth-bound: 1024 output tokens took 6.2 s at 250 W and 6.0 s
  at 300 W.
- Both cards always get the same cap. Under TP=2 the cards work in lock-step, so the
  lower-capped card sets the pace of the pair.
- Two full prefill ladders, 2.56M prompt tokens each, at 200/200 W and 250/250 W: no
  power-off, no Xid error, no reboot.
- MTP speculative decoding (separate run, different model): 2.1x decode speed at
  temperature 0, 1.6-1.7x at the temperatures an agent actually uses (0.6-1.0).

## Hardware and software

| | |
|---|---|
| GPUs | 2x NVIDIA RTX 3090 24 GB (Ampere, no native FP8), NVLink bridge |
| Host | AMD Ryzen 9 5950X, B550 board, PCIe 4.0 x8/x8, 128 GB DDR4-3600 (2 channels) |
| PSU | 1300 W |
| Serving | vLLM OpenAI-compatible server, nightly image `vllm/vllm-openai:cu129-nightly` (0.21 line; the exact build of the test day was not recorded) |
| Model | `QuantTrio/Qwen3.6-35B-A3B-AWQ`: Qwen3.6-35B-A3B MoE (35B total, ~3B active), AWQ 4-bit |
| Server flags | `--tensor-parallel-size 2`, `--max-model-len 262144`, chunked prefill with 4096-token chunks, prefix caching on |
| Power control | power caps set by a GPU-control daemon (LACT), always equal on both cards |

The launch command is in [`scripts/serve_vllm_tp2.sh`](scripts/serve_vllm_tp2.sh).

## Method

- **Prompts.** Random pseudo-words, a fresh text for every request, so the prefix cache
  never hits and every request is a full prefill. Prompt length is set exactly by
  calling the server's `/tokenize` endpoint and adjusting the word count.
- **Metric.** `max_tokens=16`, streaming; time to first token (TTFT) is taken as prefill
  time. Prefill throughput = prompt tokens / TTFT.
- **Telemetry.** `nvidia-smi` at 1 Hz: power draw, limit, temperature, clocks, throttle
  reasons. A 1 Hz log does not see sub-second transients, so for the PSU the detector is
  whether the machine stays up.
- **Ladder.** Five stages, the same at every cap:

  | stage | load |
  |---|---|
  | S1 decode | 8 requests, 24-token prompt, 1024 output tokens |
  | S2 bursts | 2 x 20K, 2 x 80K, 2 x 160K prompt tokens |
  | S3a full | 132K, 207K, 248K prompt tokens |
  | S3b campaign | 8 x 131K back to back, no pause |
  | S3c parallel | 3 x 131K sent at once |

  The driver is [`scripts/prefill_ladder.py`](scripts/prefill_ladder.py), telemetry is
  [`scripts/gpu_telemetry.sh`](scripts/gpu_telemetry.sh).
- **Caps.** 200/200 W and 250/250 W ran the full ladder. 300/300 W was a one-off ~4.5 minute
  measurement (3 decode requests, 2 x ~200K and 2 x 132K prompts) and was reverted
  right after; it is outside the range the machine runs in.

## Results

### Prefill and decode vs power cap

| metric | 200 W | 250 W | 300 W | 250 -> 300 W |
|---|---:|---:|---:|---:|
| prefill, 132K prompt, tok/s | 3272 | 4156 | 4344 / 4322 | +4-5 % |
| prefill, ~200K prompt, tok/s | 2342 (207K) | 2996 (207K) | 3342 / 3272 (199.5K) | +10 % |
| decode, 1024 tokens, s | 1.8-8.5 (noisy) | 6.2 | 6.0 / 2.6 / 5.9 | no change |
| peak power gpu0 / gpu1, W | 200.7 / 202.0 | 252.8 / 251.2 | 299.5 / 299.9 | |
| max temperature gpu0, °C | 82 | 87 | 88 | |

The 300 W ~200K points are 199.5K prompts, compared with the 207K points at 200 and 250 W.
The 200 W decode numbers were too noisy to compare with 250 W; the decode conclusion
rests on 250 vs 300 W.

### Sustained ladder, 200/200 W and 250/250 W

| stage | TTFT at 200 W, s | TTFT at 250 W, s | max °C gpu0 / gpu1 at 200 W | at 250 W |
|---|---|---|---|---|
| S2 bursts 20K / 80K / 160K | 2.8-3.0 / 18.0-18.1 / 56.3-57.5 | 2.2 / 14.2 / 44.3 | 78 / 61 | 83 / 65 |
| S3a 132K / 207K / 248K | 40.3 / 88.4 / 123.8 | 31.8 / 69.0 / 96.5 | 81 / 63 | 84 / 66 |
| S3b 8 x 131K, no pause | 40.2-45.9 | 32.4-34.0 | 82 / 64 | 87 / 67 |
| S3c 3 x 131K in parallel | 44.8 / 89.5 / 134.5 | 34.0 / 67.5 / 100.8 | 82 / 63 | 87 / 66 |

- 2.56M prompt tokens per ladder, all requests completed. About 14.8 minutes of
  continuous prefill at 200 W, ~6 of them with no pause at all (S3b).
- Single-request prefill ~3.3K tok/s at 131K; ~2.9K tok/s aggregate with three in parallel.
- Throttling in the daemon logs: only `SW_POWER_CAP`, i.e. the card sitting on its cap.
  The thermal clamp (88 °C) did not fire in either ladder.
- Zero power-offs, zero Xid errors, one boot for the whole evening. At 250 W a 49 GB
  model download ran in the background during the ladder.

Full per-stage table: [`results/ladder_ttft.csv`](results/ladder_ttft.csv).

### Speculative decoding (MTP)

A separate run on the same two cards, at 200/200 W, with a different model and runtime:
the hybrid checkpoint `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` (experts and linear
layers int4 W4A16, MoE expert offload to system RAM) served by the community
vLLM-based runtime `DominikBucko/qwen38-flash-next-2x3090` v0.5.0, MTP with 3 draft tokens.
The same profile was started a second time without a speculative config.

| mode | decode, 131K context, T=0 | decode, 32K of code, T=0 | agent probe ~29K, T=0.6-1.0 | prefill 131K |
|---|---:|---:|---:|---:|
| MTP, 3 draft tokens | 106.9-108.3 tok/s | 79.7-85.1 | 82-91 | 2323-2345 tok/s |
| no speculative decoding | 50.6-50.7 | 50.6-52.2 | 48.2-54.1 | 2257-2259 |
| speedup | x2.1 | x1.6 | x1.6-1.7 | x1.03 |

- Mean accepted length (vLLM metrics): 3.98 of 4 at T=0, ~3.2 (2.5-3.65) at T=0.6-1.0.
  A temperature-0 benchmark overstates what an agent sees.
- Tool-call choice and JSON validity in the agent probe were the same with and without
  MTP (8/8 valid JSON in both).
- On this runtime 250 W gave only +3 % prefill: here the prefill bottleneck is moving
  offloaded experts over PCIe x8, not GPU power.

The decode probe in [`scripts/decode_bench.py`](scripts/decode_bench.py) reproduces the
measurement shape (decode tok/s excluding prefill, several temperatures, fresh prefix per
request). The numbers above were taken with the runtime author's own benchmark script.

## Conclusions

1. **Prefill is limited by power, decode by memory bandwidth.** 200 -> 250 W buys +27 %
   prefill at 132K; 250 -> 300 W buys 4-5 % (10 % at ~200K, where attention is a larger
   share). Decode does not move between 250 and 300 W.
2. **Same cap on both cards.** With TP=2 every layer waits for the slower card. When this
   machine earlier ran one card at 180 W and the other higher, compute-heavy jobs ran about
   2-3x slower than at equal caps. Equal caps also make the combined draw predictable for
   the PSU. Caps here are only ever set to one of {180, 200, 230, 250} W, on both cards at
   once ([`scripts/set_power_parity.sh`](scripts/set_power_parity.sh),
   [`scripts/verify_power_parity.sh`](scripts/verify_power_parity.sh)).
3. **The ceiling is thermal, not electrical.** At identical 200 W the top card runs 18 °C
   hotter than the bottom one (83 vs 65 °C under the same load). At 250 W it reached 87 °C
   against an 88 °C clamp; at 300 W it touched 88 °C within minutes. So 200 W is the
   default, 250 W is opt-in for long-context jobs, 300 W was not adopted.
4. **Scale power to the prefill share of the job.** On a short-input, long-output
   evaluation run, prefill was 0.41 % of wall time (1.47 % with 8 clients), so +25 % prefill
   speed would return 0.08-0.29 % of total runtime. 250 W pays for long-context
   campaigns; for decode-heavy work it only adds heat.

## Repository layout

```
scripts/
  serve_vllm_tp2.sh        launch vLLM, TP=2, flags of the study
  prefill_ladder.py        ladder driver: exact-length random prompts, TTFT per request, JSONL
  decode_bench.py          decode tok/s probe, several temperatures
  gpu_telemetry.sh         nvidia-smi 1 Hz CSV logger
  set_power_parity.sh      set one equal power cap on every GPU, refuse values outside the set
  verify_power_parity.sh   PASS/FAIL check that caps are equal and allowed
results/
  *.csv                    summary tables (see results/README.md)
```

## Reproducing

```bash
# 1. serve
MODEL_DIR=/path/to/hf-cache ./scripts/serve_vllm_tp2.sh

# 2. set the cap on both cards (skip if a daemon owns power limits; use the daemon)
sudo ./scripts/set_power_parity.sh 200 && ./scripts/verify_power_parity.sh

# 3. telemetry in one terminal, ladder in another
./scripts/gpu_telemetry.sh power-200w.csv
python3 scripts/prefill_ladder.py --base-url http://localhost:8000 --out ladder-200w.jsonl

# 4. repeat at 250 W
```

The scripts need only Python 3.9+ (stdlib), bash and `nvidia-smi`.

## Raw data

Raw logs (1 Hz power and temperature, per-request timings) stayed on the test machine and
are not published. The tables above and the CSVs in `results/` are transcribed from the
written test reports made from those logs on the day of each run.

## License

MIT, see [LICENSE](LICENSE).
