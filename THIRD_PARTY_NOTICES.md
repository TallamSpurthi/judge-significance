# Third-party notices

This repository's own code is under the MIT license ([LICENSE](LICENSE)). The material below comes from other
projects and keeps its own license.

## UMBRELA prompt templates (Apache-2.0)

`src/judge_significance/llm_label.py` contains two prompt templates copied verbatim from UMBRELA
(github.com/castorini/umbrela):

| Constant in `llm_label.py` | Source file in UMBRELA |
|---|---|
| `UMBRELA_BING` | `src/umbrela/prompts/prompt_templates/qrel_zeroshot_bing.yaml` |
| `UMBRELA_BASIC` | `src/umbrela/prompts/prompt_templates/qrel_zeroshot_basic.yaml` |

UMBRELA is licensed under the Apache License, Version 2.0. A copy of its license file, including its copyright
line ("Copyright 2019-2021 Pyserini authors"), is in [LICENSES/Apache-2.0-umbrela.txt](LICENSES/Apache-2.0-umbrela.txt).
The UMBRELA repository has no NOTICE file. Both constants are unchanged copies. One judge prompt appends a separate
line to `UMBRELA_BING`, the constant `UMBRELA_ADDED`, marked `# ADDED` in `llm_label.py`; `UMBRELA_BASIC` is used as is.

## LLMJudge sample prompt (not included)

The judge haiku-llmjudge used the sample prompt of the LLMJudge repository (github.com/llm4eval/LLMJudge,
`sample-prompt.txt`, commit 2274ad6) followed by one added line, `LLMJUDGE_ADDED`, marked `# ADDED` in
`src/judge_significance/llm_label.py`. That repository states no license, so its text is not copied here. The code
reads it from the path in `LLMJUDGE_PROMPT` and checks its SHA-256 (README, Data).

## Data

The LLMJudge labels, the NIST qrels and the TREC runs are not stored in this repository and keep their own terms;
see the README, Data.
