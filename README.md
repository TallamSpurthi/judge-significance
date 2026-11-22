# judge-significance

Significance tests for comparisons between relevance judges. Judge benchmarks rank LLM relevance judges by Kendall's τ between the system ranking a judge induces and the ranking under human labels. This package tests whether the τ difference between two judges is distinguishable from topic sampling noise, with two tests whose null hypotheses differ: a paired topic-permutation test (judges exchangeable within each topic) and a bootstrap z-test (equal τ), with Holm and Benjamini–Hochberg correction. It also measures bootstrap-interval coverage and test power against known targets.

The main application is the LLMJudge benchmark (Rahmani et al., *Judging the Judges: A Collection of LLM-Generated Relevance Judgements*, arXiv:2502.13908). The code reproduces its Table 3 exactly before testing it. A replication uses TREC DL 2019 and 2020 with ten LLM judge configurations from Arabzadeh & Clarke (SIGIR 2025) and Farzi & Dietz (SIGIR 2025).

## Results at a glance

Full tables, with the script and log behind each value: [results/README.md](results/README.md).

- **Reproduction:** all 33 published τ of LLMJudge's Table 3 are exact at 4 decimals with the inferred set of 31 runs, as are Spearman's ρ and, with labels clipped to [0, 3], Cohen's κ.
- **Leaderboard:** 0 of 32 adjacent judge pairs are separable under the permutation test or the bootstrap z-test. The top judge differs from 7 of 32 others under the permutation test and 1 of 32 under the bootstrap z-test before correction, and from none after Holm or Benjamini–Hochberg.
- **Resolution** (bootstrap z-test, nominal α = 0.05): none below 200 topics, 0.116 at 200, 0.087 at 400. Calibrated on the exactly tied pairs together: none at 200, 0.116 at 400.
- **Exactly tied judge pairs** (9) are rejected in a median of 0.13 of samples at 25 topics and 0.24 at 400 under the permutation test, and 0.03 and 0.11 under the bootstrap z-test.
- **Single-judge intervals** under-cover: median coverage 0.54 (percentile), 0.74 (BCa) and 0.89 (studentized) against 0.95.
- **TREC DL 2019 and 2020:** no adjacent pair is separable after Holm; power predicted from 25-topic subsets is optimistic by 0.06 and 0.04 on average.

## Reproduced evaluation protocol

| Choice | Value |
|---|---|
| Effectiveness metric | nDCG@10, trec_eval semantics via pytrec_eval (linear gain; ties broken by docno) |
| Unjudged documents | Non-relevant (trec_eval default; not `-J`) |
| Judge labels | As submitted (no clipping) |
| Reference qrels | LLMJudge human test labels: 25 topics, 4,423 pairs |
| Correlation | `scipy.stats.kendalltau` (τ-b) on mean-over-topics system scores |
| Systems | 31 TREC DL 2023 passage runs: the 35 runs minus `WatS-LLM-Rerank`, `uot-yahoo_LLMs-blender`, `uot-yahoo_rankgpt35`, `uot-yahoo_rankgpt4` |

The paper does not state the run set. It was identified by exhaustive search over excluded-run sets of size k ≤ 5 ([scripts/02](scripts/02_search_run_subsets.py), [scripts/03](scripts/03_check_subset_uniqueness.py)). It is the only set that reproduces both Table 3 and Figure 1. It has not been confirmed by the benchmark organizers. The protocol is defined in [src/judge_significance/protocol.py](src/judge_significance/protocol.py).

## Repository layout

```
judge-significance/
├── src/judge_significance/   Python package
│   ├── data.py               qrels, judge labels, runs, published values
│   ├── evaluation.py         per-topic trec_eval scores (pytrec_eval)
│   ├── correlation.py        Kendall τ-a/b/c, Spearman ρ, Cohen's κ
│   ├── subsets.py            closed-form τ over all k-run exclusions
│   ├── permutation.py        permutation and studentized permutation tests, bootstrap z-test, jackknife t-test, Holm, BH
│   ├── dlbench.py            TREC DL 2019/2020 judge labels (Arabzadeh & Clarke; Farzi & Dietz)
│   ├── bootstrap.py          paired topic bootstrap, percentile and BCa intervals
│   ├── simulation.py         synthetic judges; percentile, BCa and studentized intervals
│   ├── llm_label.py          prompts, request bodies, output parsing and cost for the new LLM judges
│   └── protocol.py           the reproduced Table 3 protocol
├── scripts/                  numbered analysis steps, run_g_j1.sh, run_g_j2.sh, run_g_j3.sh
├── resources/                published reference values (Table 3)
├── results/README.md        every result, with its script and log
├── results/g-j1/             outputs of scripts/run_g_j1.sh (reproduction, permutation tests)
├── results/g-j2/             outputs of scripts/run_g_j2.sh (bootstrap, coverage, power)
├── results/g-j3/             outputs of scripts/run_g_j3.sh (TREC DL 2019/2020, robustness checks)
├── results/llm-judges/       new LLM judges: labels, power and summary (steps 27-31 and 36; see its README)
├── tests/                    unit tests (no data needed)
├── archive/pilot/            superseded smoke-test pilot (Nov 8, 2026)
├── LICENSES/                 third-party license texts (UMBRELA, Apache-2.0)
└── THIRD_PARTY_NOTICES.md    what is copied from other projects, and under which license
```

## Installation

Python 3.10 or later.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e '.[figures]'
.venv/bin/python -m pytest tests
```

The reproduction tests in `tests/test_reproduction.py` check every reproduction claim against the data and are skipped unless `LLMJUDGE_DATA` is set (about 25 seconds):

```
LLMJUDGE_DATA=/path/to/data .venv/bin/python -m unittest tests.test_reproduction -v
```

They assert that: an independent pure-Python nDCG@10 matches pytrec_eval for every run, topic and label set; the 31-run protocol reproduces all 33 τ and ρ (also with the independent scorer) and κ with clipped labels; both Figure 1 values reproduce; all 35 runs do not; exactness requires trec_eval's single-precision scores (full-precision ranking gives 27 of 33); the exhaustive search over all 384,168 exclusions of up to five runs finds exactly two sets matching all 33 τ, only one of which also matches Figure 1, and no other exclusion size matches more than 1 of 33; and the conclusion (no adjacent pair separable; none of the top-vs-other comparisons after Holm) is the same under both candidate sets and all 35 runs.

Two more test files have tests that need data: `tests/test_new_judges.py` (also `LLMJUDGE_DATA`, with the NIST DL 2023 qrels) and `tests/test_dlbench.py` (`DLBENCH_DATA`, see Data). With both variables set, `pytest tests` skips nothing and takes about 40 seconds.

`requirements.txt` pins the exact versions used for the reported results (Python 3.13, macOS).

## Usage

Regenerate every result, in this order. On an Apple M4 Pro laptop (12 cores), `run_g_j1.sh` takes about 4 minutes, `run_g_j2.sh` roughly 6 to 7 hours with `W=6` worker processes (the coverage simulations, steps 9 and 15, take most of it) and `run_g_j3.sh` about 45 minutes:

```
PYTHON=.venv/bin/python scripts/run_g_j1.sh DATA_DIR
PYTHON=.venv/bin/python W=6 scripts/run_g_j2.sh DATA_DIR     # also writes the figures, numbers.tex and tables/permutation.tex (not in the paper)
PYTHON=.venv/bin/python scripts/run_g_j3.sh DLBENCH_DIR      # TREC DL 2019/2020; also writes the three-benchmark table and the appendix tables and figures (steps 37-40)
```

Generated tables, figures and `numbers.tex` go to `paper/` inside this repository (git-ignored): `paper/manuscript/` and, for the ACM AI Letters version, `paper/venues/acm-ai-letters/`. Set `PAPER_ROOT` to write under another folder; a relative path is taken from the repository root.

Or run steps one at a time, from the repository root:

| Step | Script | Output |
|---|---|---|
| 1 | `01_reproduce_table3_grid.py` | κ check; Figure 1 on 35 runs; 1,512-cell protocol grid |
| 2 | `02_search_run_subsets.py --kmax 4` | Run-set search against Figure 1 and Table 3 |
| 3 | `03_check_subset_uniqueness.py --kmax 5` | Uniqueness of the run set; ρ check |
| 4 | `04_compare_table3.py` | Published vs reproduced τ and ρ per judge |
| 5 | `05_check_figure1.py` | Figure 1 under the 31-run protocol |
| 6 | `06_permutation_tests.py --perms 10000 [--seed S] [--alt-runset] [--tsv FILE]` | Adjacent-pair and top-vs-all permutation tests |
| 7 | `07_make_permutation_table.py` | LaTeX permutation table (paper/manuscript/tables/permutation.tex, not used in the paper), from the step-6 and step-18 TSVs (run after step 18) |
| 8 | `08_bootstrap_check.py --boot 10000 [--seed S] [--tsv FILE]` | Percentile and BCa intervals for τ and adjacent Δτ |
| 9 | `09_coverage_simulation.py --reps 1000 --boot 1000 --sizes 25 15 [--shared-eps 0.3]` | Coverage of percentile, BCa and studentized intervals (synthetic judges, 82-topic target) |
| 10 | `10_power_curve.py --reps 200 --perms 500 [--shared-eps 0.3] [--test permutation\|bootstrap\|jackknife]` | Rejection rate vs number of topics, by Δτ (synthetic judges); `--shared-eps` gives judges a shared error component |
| 11 | `11_disagreement_analysis.py` | System pairs that LLMJudge judges disagree about, and whether the human labels separate them |
| 12 | `12_make_figures.py` | Paper figures (real-judge coverage in the matched design and power; the without-replacement coverage as `coverage_without_replacement.pdf`; synthetic versions as `*_synthetic.pdf`) into paper/manuscript/figures/; `--only coverage power` rewrites just the coverage and power figures. The leaderboard, coverage and power figures use the paper style (Linux Biolinum from TeX Live, converted to TrueType at run time; black text; missing glyphs are errors) |
| 14 | `14_power_mechanism.py` | What synthetic judge pairs disagree about, vs their rejection rates (run before step 13) |
| 15 | `15_real_judge_coverage.py --sizes 10 15 20 --workers W`; `--sizes 25 --replace --tau-only` | Coverage of each real judge's published τ when 10, 15 or 20 of the 25 topics are drawn without replacement, and in the matched design (25 drawn with replacement, `15_real_judge_coverage_matched.tsv`) |
| 16 | `16_real_judge_power.py --test permutation\|bootstrap --workers W` | Rejection rate for all 528 real judge pairs at 25–400 topics; smallest detectable gap; `--pvalues` also saves each replicate's p-value (for step 32) without changing any draw |
| 17 | `17_error_placement.py` | Leaderboards re-drawn with each judge's error rates fixed but its errors placed on other documents |
| 18 | `18_equal_tau_tests.py --overlap-tsv FILE` | Bootstrap z-test, studentized intervals, Holm and BH on LLMJudge; double-precision ranking; judge label overlap |
| 19 | `19_dl_benchmark.py` | TREC DL 2019/2020 replication: validation, leaderboard, both tests, power (needs `DLBENCH_DATA`, see Data) |
| 20 | `20_make_replication_table.py` | LaTeX table of the three benchmarks |
| 21 | `21_stratified_check.py` | Bootstrap results with topics resampled within query type (15 real, 10 generated) |
| 22 | `22_wald_coverage.py` | Coverage of the interval the bootstrap z-test inverts, for real adjacent pairs |
| 23 | `23_test_size_check.py` | Rejection rates of the permutation, studentized permutation and bootstrap z-tests on exactly tied and widely separated real judge pairs |
| 24 | `24_extrapolation_check.py` | TREC DL 2019/2020: power predicted from 25-topic subsets vs measured on all topics; resolution with 31 vs 59 systems |
| 25 | `25_leaderboard_figure.py` | Paper figure: each judge's gap to the top judge with uncorrected and simultaneous bootstrap intervals; `--plot-only --tsv results/g-j2/25_leaderboard_intervals.tsv` redraws the leaderboard figure from the cached intervals without recomputing; add `--label-size 7 --height 3.6 --legend-above` for the ACM AI Letters version |
| 26 | `26_kappa_separability.py` | Separability of the judges by label-level agreement (Cohen's kappa), with topic-bootstrap tests |
| 27 | `27_extract_passages.py` | Streams the official MS MARCO v2 passage tar (MD5-checked, nothing stored) and keeps the judged DL 2023 passages |
| 28 | `28_llm_label.py` | Labels query-passage pairs with eight LLM judges (Claude via Message Batches or synchronously; open-weight models via OpenRouter); pilot gates, cost projection, parse checks |
| 29 | `29_claude_judge_power.py` | Power of each pair of new judges, predicted from the 25 LLMJudge topics and measured on the 57 held-out DL 2023 topics |
| 30 | `30_claude_judge_summary.py` | Summary for the paper and the released labels in `results/llm-judges/` (see its README) |
| 31 | `31_claude_retest.py` | Run-to-run variation of two Claude judges on the 25 LLMJudge topics, and the judge order with either run |
| 32 | `32_size_adjusted_resolution.py` | Bootstrap-test resolution with the test calibrated on the exactly tied pairs (pooled, and every tied pair held to 5%), from the p-values that step 16 saves with `--pvalues` |
| 33 | `33_calibrated_tied_rejection.py` | Replays step 23 exactly for the 9 exactly tied pairs (same seeds), saves their p-values (`results/g-j2/33_calibrated_tied_rejection.npz`) and reports the share rejected at step 23's calibrated cutoffs: 6.44% and 6.28% (permutation, 25 and 100 topics), 5.89% and 5.44% (studentized), 5.00% (bootstrap z) |
| 34 | `34_robustness_checks.py` | Robustness checks on saved outputs (no new simulation): held-out reversals pair by pair, calibration transfer leaving one tied-pair group out, DL 2019's tied pair, resolution by criterion (see [results/README.md](results/README.md)) |
| 35 | `35_make_resolution_table.py` | LaTeX table of the bootstrap z-test's resolution on LLMJudge by number of topics (nominal, pooled calibration, every tied pair held to 5%), from step 32's `32_size_adjusted_resolution.tsv`; no new simulation |
| 36 | `36_heldout_figure.py` | Paper figure: the eight new judges' tau on the 25 LLMJudge topics and the 57 held-out topics (values, labelled at both ends) with a bar as long as the resolution at 200 topics; writes `results/llm-judges/36_tau_by_pool.tsv` from step 29's table; no new simulation; add `--width 4.2` for the ACM AI Letters version |
| 37 | `37_make_reproduction_table.py` | ACM AI Letters appendix table: each public judge's published tau, our tau_b with the inferred 31-run set and with all 35 runs, from step 4's `04_table3_comparison.log`; standard library only; no new computation |
| 38 | `38_make_robustness_tables.py` | ACM AI Letters appendix: resolution under four criteria, calibration transfer with one group of tied pairs left out (two LaTeX tables) and macros for the held-out reversals pair by pair, from step 34's `34_robustness_checks.log`; standard library only; no new simulation |
| 39 | `39_appendix_figures.py` | ACM AI Letters appendix figures, drawn from released results (no new simulation): per-pair power against the gap in published τ at 200 and 400 topics, with the resolution (steps 16 and 32), and predicted against held-out power for the 28 pairs of new judges (step 29); checks the numbers the paper states about these data before saving |
| 40 | `40_make_test_tables.py` | ACM AI Letters appendix tables: the 32 adjacent pairs of the published order with both tests' p (step 18, permutation p checked against step 6), and the three tests on the same samples (step 23); checks the paper's macros in `numbers.tex` before writing; standard library only |
| 13 | `13_make_numbers.py` | LaTeX macros for every number in the paper, with sources, into paper/manuscript/numbers.tex and numbers.md |

## Data

No data are stored in this repository. `DATA_DIR` must contain:

| Path | Contents | Source |
|---|---|---|
| `qid_map.txt`, `docid_map.txt` | LLMJudge query and passage ID maps | github.com/llm4eval/LLMJudge, `data/ids-mapping-files/` |
| `human_ids.txt` | Human test qrels (25 topics, 4,423 pairs) | github.com/llm4eval/LLMJudge-benchmark, `llmjudge/qrels/llmjudge_test_qrel_ids.txt` |
| `judges/<name>.txt` | 33 public judge label sets | LLMJudge-benchmark, `submissions/scores/` |
| `dev_qrels_idx.txt` | Dev qrels (Figure 1 check) | LLMJudge, `data/llm4eval_dev_qrel_2024.txt` |
| `dl23/2023.qrels.pass.withDupes.txt` | Full NIST DL-2023 passage qrels (Figure 1 check) | trec.nist.gov/data/deep/ |
| `runs/<run>` | 35 TREC DL 2023 passage runs | trec.nist.gov/results, the official source (NIST data agreement) |

The official source for all TREC runs used here, DL 2019, 2020 and 2023, is NIST (trec.nist.gov/results, under its data agreement). The runs can be kept outside `DATA_DIR`: set `LLMJUDGE_RUNS` to a folder of DL 2023 run files, or `DLBENCH_RUNS` to a folder with `dl19/` and `dl20/` subfolders. Run files may be plain or gzipped; a run's name is its file name without an `input.` prefix and a `.gz` or `.txt` suffix, so NIST's `input.<run>.gz` files can be used as downloaded.

The LLMJudge sample prompt is not stored here either, because the LLMJudge repository states no license. Labelling new pairs with the judge haiku-llmjudge (step 28) needs `LLMJUDGE_PROMPT` set to your copy of `sample-prompt.txt` from github.com/llm4eval/LLMJudge (commit 2274ad6); the code checks its SHA-256. Nothing else needs it: the released labels and every number in the paper can be reproduced without it.

The LLMJudge human test qrels are a subsample of the full NIST qrels (4,423 of 6,427 pairs on the 25 test topics, with identical labels on every shared pair).

The TREC DL 2019/2020 replication (step 19) reads a second directory, `DLBENCH_DATA` (default `~/data/dlbench`), with sparse checkouts of github.com/Narabzad/llm-relevance-judgement-comparison (`raw qrels/`, `data/`, `runs/`) and github.com/TREMA-UNH/appendix-umbrella-other-llm (`result logs/`, `data/`). Neither repository states a licence; cite both papers. The first repository also holds copies of the DL 2019 and 2020 runs, which the code reads by default; the official files are at NIST (set `DLBENCH_RUNS` to use them). The per-topic human nDCG@10 files (`runs/dl19.ndcg_cut10`, `runs/dl20.ndcg_cut10`) are always read from that repository. See `src/judge_significance/dlbench.py` for the exact files and label rules.

## Known limitations

- The run set is inferred, not documented; see above.
- The results use a public mirror of the DL-2023 runs, not the official NIST files. That the mirror's runs equal NIST's is unchecked; the exact reproduction of LLMJudge's Table 3 is the only evidence.
- The DL 2019/2020 results use the copies of the runs in the Arabzadeh & Clarke repository, not the official NIST files. That they are equal is likewise unchecked; the reproduction of the published human nDCG@10 is the only evidence.
- Real-judge coverage and power treat the 25 LLMJudge topics as the population; power above 25 topics extrapolates their distribution. Synthetic judges on 82 topics and the DL 2019/2020 replication are checks with larger populations.
- Power summaries leave out six near-duplicate judge pairs, whose labels differ on fewer than 1% of pairs (`results/g-j2/18_label_overlap.tsv`, `label_diff_share < 0.01`): NISTRetrieval-instruct0, -instruct1 and -instruct2 pairwise, and NISTRetrieval-reason0, -reason1 and -reason2 pairwise. Of the 13 pairs with exactly equal published τ, 4 are among them, which leaves the 9 tied pairs reported in the paper.
- The DL 2019 and 2020 rows of the three-benchmark table use the Farzi & Dietz labels as released. Re-parsing the llama-3.3-70b/umbrela labels from the model output (step 19) is a sensitivity check only; it moves that configuration's DL 2020 τ from 0.878 to 0.910 (published 0.911).
- No test here is exactly calibrated for "equal τ": the permutation test over-rejects equal-τ judges that differ in per-topic noise, and the bootstrap z-test drifts above nominal at 400 topics.

## License

Code: MIT ([LICENSE](LICENSE)). The license covers this repository's code and the derived values in `resources/` and `results/` only. It does not cover the LLMJudge labels, the NIST qrels or the TREC runs, which keep their own terms (see Data). Two UMBRELA prompt templates in `src/judge_significance/llm_label.py` are Apache-2.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/`.
