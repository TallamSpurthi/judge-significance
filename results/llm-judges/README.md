# New LLM judges on TREC DL 2023 (steps 27-30)

Labels from eight LLM judges for nearly every NIST-judged query-passage pair of the 82 TREC DL 2023 topics (22,327 pairs; `summary.tsv` has the count labelled by each judge, 22,292 at least), with the paper's held-out power analysis.

**Please do not use these labels to train or tune models.** TREC DL 2023 is a public test collection, and labels in training data would inflate a model's agreement with the TREC labels on it.

## Files

- `qrels/<judge>.txt`: TREC qrels format (`qid 0 passage_id grade`), grades 0-3. These are ids and grades only; there is no passage text. A pair a judge could not label after one retry is absent; absent pairs are scored as non-relevant (pytrec_eval default, `judged_docs_only_flag=False`).
- `29_power_bootstrap.tsv`: step 29 output. For each topic pool and judge pair: tau of both judges, the gap, and bootstrap z-test power at 25, 50, 100 and 200 topics (500 replicates, B = 300, seed 0).
- `36_tau_by_pool.tsv`: step 36 output: each new judge's tau on the 25 LLMJudge topics and on the 57 held-out topics, read from `29_power_bootstrap.tsv` (the data behind the held-out figure).
- `summary.tsv`: step 30 output, the source of the paper's `\nj...` macros (via step 13).
- `retest/<judge>.txt`: a second run of haiku-basic and haiku-llmjudge on the LLMJudge test pairs (batch route; the released `qrels/` hold the first, synchronous run). Same format; 4 pairs per judge are absent.
- `retest.tsv`: step 31 output: label agreement and tau of the two runs, and the judge-order agreement between the 25 and the held-out topics with either run (source of the `\njRetest...` macros).

## Judges

| Judge | Model | Prompt | Access |
|---|---|---|---|
| haiku-umbrela | claude-haiku-5-5 | UMBRELA zero-shot plus one output-format line | Anthropic Message Batches, thinking disabled, effort low |
| haiku-llmjudge | claude-haiku-5-5 | LLMJudge sample prompt plus one output-format line | as above |
| haiku-basic | claude-haiku-5-5 | UMBRELA "basic" | as above |
| sonnet-umbrela | claude-sonnet-5-5 | UMBRELA, as haiku-umbrela | Anthropic Message Batches, thinking `between_tools` (its lowest setting), effort low |
| gemma4b-umbrela | google/gemma-3-4b-it | UMBRELA, as haiku-umbrela | OpenRouter, pinned to DeepInfra bf16 |
| llama8b-umbrela | meta-llama/llama-3.1-8b-instruct | as above | OpenRouter, pinned to DeepInfra fp8 |
| scout-umbrela | meta-llama/llama-4-scout | as above | OpenRouter, pinned to DeepInfra fp8 |
| qwen30b-umbrela | qwen/qwen3-30b-a3b-instruct-2507 | as above | OpenRouter, pinned to Nebius fp8 |

All labels were produced on November 9 and 10, 2026, with the model identifiers above. The request settings and the UMBRELA prompt texts are in `src/judge_significance/llm_label.py`, with a test that pins each text to its source. The LLMJudge sample prompt is not stored here because its repository states no license; the code reads it from your copy (`LLMJUDGE_PROMPT`, see the main README, Data) and checks that it is the text used for these labels.

**Open-weight judges** run with temperature 0 and seed 0, and with no fallback to other providers. Providers that keep prompts are refused (`data_collection: deny`), and OpenRouter's context compression is turned off. Every response records the provider and model that served it, and all matched the pinned ones.

**Claude models** accept no sampling parameters, so their labels are not deterministic. On the same 4,423 pairs, two runs of haiku-llmjudge agreed on 91.8% of labels, and two runs of haiku-basic on 73.3%. On the 25 topics this moves their tau by 0.014 and 0.030 (step 31); with its second run haiku-basic would rank first of the eight judges instead of sixth. Pairs the second run lacks do not change any tau (checked with `--fill`).

**One cut passage.** Three open-weight judges (gemma4b, llama8b and scout) have a 131,072-token context. For them, passages longer than 280,000 characters were cut to their first 280,000 characters. This affects one 451,714-character passage, judged for four topics, so 4 of 22,327 pairs. All other judges saw every passage whole.

**Labelling route.** The 4,423 LLMJudge test pairs were labelled in a pilot. haiku-llmjudge and haiku-basic went through the synchronous API there, because their batches stayed queued; the other judges used batches. The remaining 17,904 pairs were labelled by batch (Claude) or synchronously (OpenRouter). Every unparsable output was retried once.

**Cost:** $33.93 Anthropic and $4.69 OpenRouter, $38.62 in total.

## Reproduce

```
python scripts/27_extract_passages.py ...      # passage text from the official MS MARCO v2 tar (MD5-checked)
python scripts/28_llm_label.py pairs-pilot / pairs-main / submit / fetch / run / evaluate ...
python scripts/29_claude_judge_power.py DATA --labels DIR ... --pools test test-full heldout all --reps 500
python scripts/30_claude_judge_summary.py DATA --power results/llm-judges/29_power_bootstrap.tsv --labels DIR ... --out results/llm-judges
python scripts/31_claude_retest.py DATA --labels results/llm-judges/qrels --retest DIR --out results/llm-judges
python scripts/13_make_numbers.py
```

Steps 28-30 read API keys from the environment (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) and never write them anywhere. Step 28 with haiku-llmjudge also needs `LLMJUDGE_PROMPT`. Steps 29 and 31 can be rerun from the released labels without any API access: pass `--labels results/llm-judges/qrels` (and, for step 31, `--retest results/llm-judges/retest`).
