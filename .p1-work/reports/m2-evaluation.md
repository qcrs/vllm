# M2 Quality and Serving Evaluation

## Environment and scope

All model runs use `/data/models/Qwen3-8B` BF16, FLASH_ATTN/B16, MRV2,
TP1, physical GPU1, eager execution, and the vLLM v0.26.0 baseline. GPU runs
are serialized with `.p1-work/scripts/run_guard.py`; the project-wide memory
limit is 45 GiB and the guard stops at 44 GiB. Full source invocations,
per-run source snapshots, NVML samples, and exit status are retained under
`/data_ssd/qcrs/experiments/p1-v8/M2`.

## Matched quality sample

The exploratory set contains 85 audited first-turn records: 25 each from
BabiLong 8K, 16K, and 32K, eight SCBench many-shot records, and two SCBench
Chinese QA records. Every policy uses the same manifest and selected rows
(manifest SHA-256 `967d4e1d7e321ddd85c65f0081575b1b0573f71c69a9273e549e3e67e023c26b`,
selection SHA-256
`5b8d762770f0b3770af0b0a167741189d8405313723e93553b902f146b5310c6`). Source
Arrow SHA, row, prompt SHA, rendered chat prompt, and token count were checked
for every sample. The protocol forces the same newline token after full
prefill, then clears the sampling mask, runs a q=0 maintenance step for
reclamation policies, and scores the free answer. Generation is deterministic
(temperature 0, seed 0, 128-token cap).

| Policy | Target contained / 85 | Mean whitespace-token F1 | Paired losses / gains vs Full | Released pages per reclaimed sample | Maintenance host p50 / p95 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 45 | 0.1973 | baseline | 0 | n/a |
| SinkRecent | 41 | 0.1737 | 8 / 4 | 191-985 | 177.0 / 188.6 / 196.5 |
| Random | 39 | 0.1430 | 11 / 5 | 191-985 | 6.9 / 164.8 / 177.1 |
| TriAttention-inspired | 44 | 0.2036 | 1 / 0 | 191-985 | 14.9 / 18.5 / 372.4 |

All four policies scored 0/2 on SCBench Chinese QA. BabiLong normalized exact
match was zero except SinkRecent and TriAttention-inspired at 32K (1/25 each).
Three, four, ten, and three outputs respectively reached the 128-token cap for
Full, SinkRecent, Random, and TriAttention-inspired. The target-containment and
token-F1 measures are weak exploratory metrics for verbose English answers and
especially Chinese; this sample does not establish quality preservation or
justify enabling lossy retention by default. TriAttention's higher aggregate
F1 is not evidence of a reliable quality improvement.

Maintenance host time measures the complete `engine.step()` plus its required
completion wait, not an isolated kernel. The p99 outliers include un-attributed
work such as possible compilation and are not used as kernel latency. The
scoring kernel's independent CUDA-event/NCU results remain in
`m2-scoring.md`.

## HTTP serving comparison

`.p1-work/scripts/serving_reclamation.py` starts the OpenAI-compatible server,
warms it once, then runs three repeats of the same eight-request stream. The
4096/8192-token prompts alternate; each request generates 128 tokens at fixed
arrival offsets `[0, 0.3, 0.6, 0.9, 5, 5.3, 5.6, 5.9]` seconds. Client receipt
times come from streamed token IDs; tokens in one server chunk share a receipt
time. SLO thresholds were fixed before comparison at TTFT <= 2 s and TPOT <=
80 ms. Reported throughput includes queueing and reclamation time.

`noop` uses the `sink_recent` policy with retention 1.0. SLO-good counts are
out of eight requests per repeat. Native preemptions are per-repeat counter
deltas read from `/metrics`.

| Pool | Policy | Guard peak (bytes) | Output token/s by repeat | TTFT p99 (s) by repeat | TPOT p99 (s) by repeat | SLO-good requests by repeat | Preemptions by repeat |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| 6 GiB | Full | 23,727,177,728 | 99.12 / 99.01 / 98.76 | 1.324 / 1.335 / 1.325 | 0.040 / 0.040 / 0.038 | 8 / 8 / 8 | 0 / 0 / 0 |
| 6 GiB | Noop | 23,727,177,728 | 99.17 / 98.82 / 98.78 | 1.306 / 1.298 / 1.312 | 0.039 / 0.038 / 0.037 | 8 / 8 / 8 | 0 / 0 / 0 |
| 6 GiB | SinkRecent | 23,727,177,728 | 98.71 / 98.91 / 98.41 | 1.288 / 1.314 / 1.318 | 0.037 / 0.038 / 0.037 | 8 / 8 / 8 | 0 / 0 / 0 |
| 6 GiB | Random | 23,727,177,728 | 97.97 / 98.48 / 97.78 | 1.329 / 1.309 / 1.338 | 0.041 / 0.040 / 0.041 | 8 / 8 / 8 | 0 / 0 / 0 |
| 6 GiB | TriAttention-inspired | 23,729,274,880 | 98.26 / 97.72 / 97.13 | 1.313 / 1.348 / 1.348 | 0.038 / 0.040 / 0.040 | 8 / 8 / 8 | 0 / 0 / 0 |
| 1.5 GiB | Full | 18,907,922,432 | 9.03 / 9.01 / 8.97 | 103.00 / 103.36 / 103.82 | 0.139 / 0.139 / 0.139 | 0 / 0 / 0 | 316 / 316 / 316 |
| 1.5 GiB | Noop | 18,907,922,432 | 9.08 / 9.05 / 9.02 | 102.63 / 102.95 / 103.29 | 0.139 / 0.138 / 0.139 | 0 / 0 / 0 | 316 / 316 / 316 |
| 1.5 GiB | SinkRecent | 18,907,922,432 | 56.95 / 71.02 / 80.48 | 8.75 / 5.35 / 3.66 | 0.113 / 0.064 / 0.051 | 0 / 4 / 4 | 10 / 5 / 5 |
| 1.5 GiB | Random | 18,907,922,432 | 68.74 / 76.25 / 80.46 | 5.79 / 4.35 / 3.63 | 0.082 / 0.056 / 0.051 | 1 / 4 / 4 | 7 / 5 / 5 |
| 1.5 GiB | TriAttention-inspired | 18,910,019,584 | 66.64 / 69.84 / 71.75 | 6.13 / 5.50 / 5.01 | 0.079 / 0.071 / 0.063 | 4 / 4 / 4 | 7 / 6 / 6 |

At 6 GiB, all policies stayed below the SLO thresholds for all requests and
reported zero preemptions; strategy throughput differences are small under
this non-pressure workload. At 1.5 GiB, Full and Noop each preempted 316 times
per repeat (948 total) and had no SLO-good requests. Reclamation reduced
preemptions to 17-20 total and raised output throughput to 56.95-80.48
tokens/s, but every policy still had TTFT p99 above 3.6 s, so none met the
tail-latency target. These are pressure-probe results, not production serving
claims. The bounded-reserve behavior itself is covered by the M2 pressure
traces in `m2-pressure.md`; this serving matrix uses `reserve_cap=0`.

The SinkRecent server log records Triton JIT compilation of
`_advance_effective` and `_compact_stable` during the first measured repeat.
Treat the first-repeat latency as startup-contaminated; the three per-repeat
values are retained rather than collapsed to a single steady-state estimate.

## Limits

These tests validate the MRV2/Eager/TP1 configuration only. The synthetic HTTP
workload is a controlled pressure probe, not a representative production
traffic trace. Stream chunk receipt is not a per-token GPU timestamp. Quality
results are exploratory, and the 1.5 GiB serving results show partial relief
from native preemption without meeting the configured tail-latency SLO. The
6 GiB runs do not isolate idle scoring overhead under a separate no-pressure
arrival workload.
