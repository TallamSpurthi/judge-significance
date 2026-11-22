# Smoke-test pilot (superseded)

Written on Nov 8, 2026, before the Table 3 protocol was reproduced. Standard library only.

| File | What it does |
|---|---|
| `pilot.py` | nDCG@10 per run under human and judge qrels; τ-b per judge; percentile topic bootstrap; adjacent-pair permutation test |
| `pilot_top.py` | Same, top judge against every other judge |
| `pilot_out.txt` | Output of `pilot.py` (B = 1000, P = 1000) |

**Why it is superseded.** It uses all 35 runs (Table 3 uses 31) and exponential-gain nDCG@10 (trec_eval uses linear gain), so its τ values do not match Table 3 (0.935 vs 0.9516 for prophet-setting2). Use `scripts/` and `src/` instead. The percentile-bootstrap observation in `pilot_out.txt` has not yet been re-run under the exact protocol.

Usage: `python pilot.py DATA_DIR [B] [P]`.
