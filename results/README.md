# Results

Paths in code font are relative to the repository root.

## Reproduction and permutation tests (Nov 8, 2026)

| Check | Result |
|---|---|
| Table 3 Kendall τ (33 public judges) | 33/33 exact at 4 decimals |
| Table 3 Spearman ρ | 33/33 exact |
| Table 3 Cohen's κ | 33/33 exact (labels clipped to [0, 3]) |
| Figure 1 τ (no judges involved) | dev 0.9183 and test 0.9473, both exact |
| Adjacent judge pairs separable, paired permutation test (P = 10,000) | 0/32 (smallest p = 0.656); 0/32 after Holm |
| Top judge vs each other judge | 7/32 at uncorrected p < 0.05; 0/32 after Holm |

Logs and tables are in [g-j1/](g-j1/).

## Topic bootstrap, coverage and power (Nov 8, 2026)

Topic bootstrap under the same protocol (B = 10,000; seeds 0 and 1 agree). Logs in [g-j2/](g-j2/).

| Check | Result |
|---|---|
| Judges whose τ lies above their own 95% percentile interval | 10/33 (top judge: τ = 0.9516, interval [0.762, 0.939]) |
| Bootstrap bias, mean of replicates minus τ | median −0.062 (range −0.179 to +0.010) |
| Judges whose τ lies inside their own BCa interval | 23/33 (top judge: BCa [0.960, 0.977], above τ) |
| Adjacent-pair Δτ intervals excluding 0 | percentile 0/32, BCa 0/32 |

The percentile topic bootstrap is shifted well below τ, and BCa does not repair it: with bias-correction z0 up to 2.5, BCa moves the interval above τ for the top judges.

Coverage simulation (`scripts/09`; synthetic judges, target = τ on all 82 DL-2023 topics, 1,000 replicates per cell; 25 topics pools two judge sets, seeds 0 and 1): 95% interval coverage for τ at 25 topics is 0.02–0.99 (percentile), 0.48–0.81 (BCa) and 0.72–0.93 (studentized); at 15 topics (seed 0) 0.20–0.66, 0.55–0.75 and 0.83–0.89. For Δτ at 25 topics: 0.33–1.00, 0.75–0.90 and 0.86–0.91.

Power (`scripts/10`; 200 replicates per cell): the paired permutation test needs about 300 to more than 400 topics for 80% power at Δτ ≈ 0.05, and 17–115 at Δτ ≈ 0.10. It rejects in up to 18% of replicates when τ is equal or near-equal (gap ≤ 0.0043) but the judges differ in noise (its null is per-topic exchangeability).

## Real judges, equal-τ tests and replication (Nov 9, 2026)

The result numbers in the paper's text are generated into `paper/manuscript/numbers.tex` by `scripts/13` (see `numbers.md` beside it for each value and its source file) and, for some of the ACM AI Letters appendix, into `paper/venues/acm-ai-letters/robustness-numbers.tex` by `scripts/38`. Design constants such as sample sizes are typed in the text.

| Check | Result |
|---|---|
| LLMJudge adjacent pairs separable, bootstrap z-test (B = 10,000) | 0/32 (smallest p among nonzero gaps 0.75) |
| Top judge vs others, bootstrap z-test | 1/32 uncorrected; 0/32 after Holm or BH |
| Top judge vs others, permutation test, after BH | 0/32 (smallest BH-adjusted p 0.117) |
| Double-precision document ordering instead of trec_eval's float32 | 6 τ values change; 8 judges move, up to 4 places |
| Coverage of each real judge's published τ, matched design: 25 topics drawn with replacement (`scripts/15 --replace`; the coverage figure) | median (max): percentile 0.54 (0.99), BCa 0.74 (0.86), studentized 0.89 (0.92); one percentile interval, and no other, reaches 0.95 |
| Coverage of each real judge's published τ, 20 of 25 topics drawn without replacement (`scripts/15`; not in the paper, as this design overstates the bootstrap's variance and bias) | percentile 0.00–1.00, BCa 0.54–0.94, studentized 0.51–0.93 |
| Smallest gap 80% of real-judge pairs detect with 80% power (`scripts/16`), at nominal α = 0.05 | permutation: none at 25 or 50 topics, 0.11 at 100, 0.05 at 400; bootstrap: none below 200 topics, 0.116 at 200, 0.087 at 400 |
| Monte Carlo stability of the bootstrap resolution at 200 and 400 topics (`scripts/13`, from the step 16 powers) | Each pair's rejection count is redrawn from Binomial(200, its step 16 power), independently across pairs; 2,000 redraws, seed 13. Step 16 computes power for all 528 judge pairs; the redraw, like the resolution and every power summary in the paper, uses the 522 pairs left after the six near-duplicate pairs (labels differ on fewer than 1% of pairs; see Known limitations in the main README) are excluded. At 200 topics 1,880 redraws resolve a gap (5th–95th percentile 0.113–0.116) and 120 (6%) resolve none; at 400 all 2,000 do (0.085–0.091). Step 16 draws each pair's topics from its own seed, so no draws are shared across pairs and independent redraws match the design. The redraw treats each estimated power as the true power and covers Monte Carlo error only |
| Bootstrap resolution with the test calibrated on the 9 exactly tied pairs (`scripts/32`, from the step 16 p-values) | pooled (5% of the tied pairs' samples reject): none at 200 topics (none in 93% of Monte Carlo draws), 0.116 at 400 (none in 14%); every tied pair held to 5%: none at 200 or 400. Conditional on these 9 pairs in the 25-topic resampling design |
| Rejection rate, real-judge pairs < 0.005 apart, 25 topics | permutation 0.13, bootstrap 0.03 |
| Exactly tied real pairs (9), rejection at 25 / 400 topics (`scripts/16`) | permutation 0.13 / 0.24; bootstrap z 0.03 / 0.11 |
| Studentized permutation test (`scripts/23`) | exactly tied pairs 0.10 at 25 topics, 0.12 at 100; power for gaps ≥ 0.10 at 100 topics 0.77 (permutation 0.88, bootstrap 0.64) |
| Wald interval of the bootstrap z-test (`scripts/22`) | covers adjacent gaps 0.93–1.00 at 20 of 25 topics, 0.78–1.00 at 10 (median 0.98; the lowest pair is below the nominal 0.95) |
| Size-adjusted power, gaps ≥ 0.10, 100 topics (`scripts/23`) | permutation 0.69, studentized permutation 0.61, bootstrap z 0.59 |
| Label-level agreement (`scripts/26`) | Cohen's kappa separates the top-kappa judge from 22/32 judges (4 after Holm) and 13/32 tau-adjacent pairs (2 after Holm) |
| Extrapolation check on DL 2019/2020 (`scripts/24`) | power predicted from 25-topic subsets vs measured: correlation 0.90 / 0.93, mean error +0.06 / +0.04 (optimistic) |
| Systems vs topics, DL 2020 (`scripts/24`) | resolution at 100 topics 0.064 with all 59 systems (0.06 in the paper); with 31 systems, the median of five random 31-system draws is 0.117 (0.12 in the paper; range 0.077–0.198), and all five are coarser than the full set (`results/g-j3/24_extrapolation_check.log`, line 15; draws at `scripts/24_extrapolation_check.py` line 94) |
| Jackknife t-test (equal τ, jackknife SE) | no power (≤ 0.21 at Δτ ≈ 0.10 and 400 topics): the jackknife variance is not consistent for τ; kept for the record only |
| Error placement re-drawn with error rates fixed (`scripts/17`) | Kendall agreement with the published leaderboard 0.17–0.55; top judge lands at rank 1–29 |
| Bootstrap stratified by query type (15 real, 10 generated topics; `scripts/21`) | adjacent 0/32; top vs others 0/32 after Holm or BH; 9/33 judges above their percentile interval |
| TREC DL 2019 / 2020 (`scripts/19`; 43 and 54 topics, 10 judges) | human nDCG@10 matches the released per-topic values; 7 of 8 published τ (Farzi & Dietz) within 0.01; no adjacent pair separable after Holm |

## Robustness checks (step 34, Nov 12, 2026)

These come from saved outputs only: no new simulation, and no number in the paper changes. Reproduce with `python scripts/34_robustness_checks.py > results/g-j3/34_robustness_checks.log` (also run by `scripts/run_g_j3.sh`; it reads `results/g-j2/` and `results/g-j3/19_*`). Rejection rates are shares of 200 samples per pair and size; Monte Carlo SEs in parentheses.

**Calibration transfer, one group of tied pairs left out.** The 9 exactly tied pairs fall into 4 groups by their shared τ (identical concordance counts). For each group, the worst-pair cutoff of step 32 is fitted on the other 3 groups' step 16 p-values and applied to the held-out group (rejection pooled over its pairs) and to all 522 pairs (resolution).

| Held-out group (pairs) | 25 topics | 100 | 200 | 400 | Resolution, 200 / 400 topics |
|---|---|---|---|---|---|
| τ 0.9085 (1) | 0.005 (0.005) | 0.010 (0.007) | 0.000 | 0.010 (0.007) | none / none |
| τ 0.9042 (1) | 0.010 (0.007) | 0.000 | 0.005 (0.005) | 0.000 | none / none |
| τ 0.9009 (6 pairs among 4 judges) | 0.067 (0.007) | 0.050 (0.006) | 0.055 (0.007) | 0.072 (0.007) | none / 0.116 (11 pairs) |
| τ 0.8276 (1) | 0.005 (0.005) | 0.020 (0.010) | 0.025 (0.011) | 0.020 (0.010) | none / none |

Limits: there are only 4 groups. The 6 pairs of the largest group share 4 judges, so their samples are correlated and the SE shown, which treats 1,200 samples as independent, understates the uncertainty. That group sets the worst-pair cutoff whenever it is in the fit. Held out, the cutoff from the other three groups is looser than α at 25 topics (0.068), and the resolution at 400 topics becomes 0.116 instead of none. The worst single pair under the held-out cutoff is NISTRetrieval-reason1/TREMA-direct: 0.15 at 25 topics and 0.12 at 100, against 0.12 and 0.145 at nominal α. The paper's 0.12 and 0.15 for the worst single tied pair are this pair at nominal α, 0.145 rounded half up.

**DL 2019's exactly tied pair (out of collection; `scripts/19`).** llama-3.3-70b/umbrela vs llama3.2/umbrela, whose labels differ on 57% of pairs. Bootstrap z-test rejection: 0.015 (0.009) at 25 topics, 0.040 (0.014) at 50, 0.030 (0.012) at 100, 0.045 (0.015) at 200, 0.105 (0.022) at 400. Limits: one pair; sizes above 43 extrapolate DL 2019's 43 topics. No calibration step (16, 23, 32, 33) reads DL data.

**Resolution by criterion** (nominal α, bootstrap z-test, step 16 powers, 522 pairs):

| Share of pairs, power | 200 topics | 400 topics |
|---|---|---|
| 80%, 80% (the paper's) | 0.116 (11 pairs) | 0.087 (31) |
| 70%, 80% | 0.113 (12) | 0.070 (77) |
| 80%, 90% | none | 0.116 (11) |
| 90%, 80% | 0.116 (10) | 0.108 (17) |

Limits: the 400-topic resolution depends on the criterion (0.070 to 0.116); the 200-topic one is 0.113 to 0.116, or none under the stricter power.

**Held-out reversals, pair by pair** (`results/llm-judges/29_power_bootstrap.tsv`). Of the 10 judge pairs reversed between the 25 LLMJudge topics and the held-out topics, 4 are no longer reversed when the 25 topics are scored on all NIST-judged pairs, 3 new reversals appear, and 6 stay reversed (9 in all).
