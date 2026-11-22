"""Relevance labelling with Claude models: prompts, request bodies, output parsing, qrels I/O and cost.

Nothing here touches the network; scripts/28_llm_label.py makes the API calls.

Judges (model x prompt), all with thinking off and effort "low":
    haiku-umbrela    claude-haiku-5-5,  UMBRELA zero-shot prompt (castorini/umbrela, Apache-2.0) plus one output-format line
    haiku-llmjudge   claude-haiku-5-5,  LLMJudge sample prompt (read from LLMJUDGE_PROMPT) plus one output-format line
    haiku-basic      claude-haiku-5-5,  UMBRELA zero-shot "basic" prompt (explanation, then the label)
    sonnet-umbrela   claude-sonnet-5-5, UMBRELA zero-shot prompt
Claude Haiku 5.5 turns thinking off with {"type": "disabled"}; Claude Sonnet 5.5 rejects that and uses
{"type": "between_tools"}, its lowest setting. Neither model accepts sampling parameters, so labels are
not deterministic; raw outputs are kept for every request.
"""
import functools
import hashlib
import json
import os
import re

# --- prompts (verbatim from their sources; {query} and {passage} are filled by render()) -----------------
# Trailing spaces and dashes in the sources are written as \u escapes so that editors cannot change them.

# https://github.com/castorini/umbrela/blob/main/src/umbrela/prompts/prompt_templates/qrel_zeroshot_bing.yaml
UMBRELA_BING = """Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query,\u0020
1 = represents that the passage seems related to the query but does not answer it,\u0020
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and\u0020
3 = represents that the passage is dedicated to the query and contains the exact answer.

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

Query: {query}
Passage: {passage}

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning."""

# https://github.com/castorini/umbrela/blob/main/src/umbrela/prompts/prompt_templates/qrel_zeroshot_basic.yaml
UMBRELA_BASIC = """You are an expert judge of a content. Using your internal knowledge and simple commonsense reasoning, try to verify if the passage is relevance category to the query.
Here, "0" represent that the passage has nothing to do with the query, "1" represents that the passage seems related to the query but does not answer it, "2" represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and "3" represents that the passage is dedicated to the query and contains the exact answer.

Provide explanation for the relevance and give your answer with from one of the categories 0, 1, 2 or 3 only. One of the categorical values if compulsory in answer.

Instructions: Think about the question. After explaining your reasoning, provide your answer in terms of 0, 1, 2 or 3 category. Only provide the relevance category on the last line. Do not provide any further details on the last line.

###

Query: {query}
Passage: {passage}

Explanation:"""

# LLMJudge sample prompt: github.com/llm4eval/LLMJudge, sample-prompt.txt (commit 2274ad6). That repository has no
# license, so its text is not copied here. Set LLMJUDGE_PROMPT to your own copy of the file; llmjudge_sample() checks
# that it is the text used for the released labels (trailing blank lines dropped). One line is added to it.
LLMJUDGE_PROMPT_SHA256 = "813c8ab435e33ffabc0d044f101f1b0d508a753976daad73d3ca3dfdd2da5efe"
LLMJUDGE_ADDED = "\nRespond with the score (0, 1, 2 or 3) only."  # ADDED

# ADDED to UMBRELA_BING after the Nov 9 micro-pilot: without it, Claude Haiku 5.5 and Sonnet 5.5 follow "Split this
# problem into steps" and reason before the score (8 of 20 and 14 of 20 outputs hit the 32-token limit).
UMBRELA_ADDED = "\nRespond with the final score line only, in the format ##final score: X"  # ADDED

PROMPTS = {
    "umbrela": UMBRELA_BING + UMBRELA_ADDED,
    "basic": UMBRELA_BASIC,
}  # "llmjudge" is read from a file: template()


class PromptFileError(Exception):
    """The LLMJudge sample prompt is missing or is not the text used for the released labels."""


@functools.lru_cache(maxsize=None)
def llmjudge_sample(path):
    """The LLMJudge sample prompt from the user's copy of sample-prompt.txt, checked by SHA-256."""
    where = "sample-prompt.txt of github.com/llm4eval/LLMJudge, commit 2274ad6"
    if not path:
        raise PromptFileError(f"set LLMJUDGE_PROMPT to a copy of {where}; its text is not stored in this repository")
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            text = f.read().rstrip("\n")
    except (OSError, UnicodeDecodeError) as e:
        raise PromptFileError(f"cannot read the file in LLMJUDGE_PROMPT ({type(e).__name__}); it should be {where}") from e
    if hashlib.sha256(text.encode()).hexdigest() != LLMJUDGE_PROMPT_SHA256:
        raise PromptFileError(f"the file in LLMJUDGE_PROMPT is not {where} (SHA-256 differs)")
    return text


def template(prompt):
    """The prompt template by name ("umbrela", "basic" or "llmjudge"); only "llmjudge" needs a file."""
    if prompt == "llmjudge":
        return llmjudge_sample(os.environ.get("LLMJUDGE_PROMPT", "")) + LLMJUDGE_ADDED
    return PROMPTS[prompt]

HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
JUDGES = {
    "haiku-umbrela": dict(model=HAIKU, prompt="umbrela", thinking={"type": "disabled"}, effort="low", max_tokens=32),
    "haiku-llmjudge": dict(model=HAIKU, prompt="llmjudge", thinking={"type": "disabled"}, effort="low", max_tokens=32),
    "haiku-basic": dict(model=HAIKU, prompt="basic", thinking={"type": "disabled"}, effort="low", max_tokens=512),
    "sonnet-umbrela": dict(model=SONNET, prompt="umbrela", thinking={"type": "between_tools"}, effort="low", max_tokens=32),
}

# Open-weight judges through OpenRouter (api="openrouter"), all with the UMBRELA prompt, temperature 0 and seed 0.
# Each is pinned to one provider endpoint and precision with no fallback, so every label of a judge comes from
# the same weights; providers that keep or train on prompts are refused (data_collection "deny"), and the
# context-compression plugin, which would silently cut long prompts, is turned off. Endpoints were chosen from
# GET /api/v1/models/{model}/endpoints on Nov 9, 2026.
OPEN_JUDGES = {
    "gemma4b-umbrela": dict(model="google/gemma-3-4b-it", provider="deepinfra", quantization="bf16"),  # Parasail rate-limited (micro-pilot)
    "llama8b-umbrela": dict(model="meta-llama/llama-3.1-8b-instruct", provider="deepinfra", quantization="fp8"),
    "qwen30b-umbrela": dict(model="qwen/qwen3-30b-a3b-instruct-2507", provider="nebius", quantization="fp8"),
    "scout-umbrela": dict(model="meta-llama/llama-4-scout", provider="deepinfra", quantization="fp8"),
}
for _j in OPEN_JUDGES.values():
    _j.update(api="openrouter", prompt="umbrela", max_tokens=32, temperature=0.0, seed=0, max_passage_chars=None)
# The three endpoints with a 131,072-token context cannot take the longest DL 2023 passage (451,714 characters,
# 168,472 Llama 3.1 tokens); Gemma 3 used 2.45 characters per token on the longest pilot passage, so 280,000
# characters (about 114K tokens) fits. This cuts 4 of the 22,327 judged pairs (one document, four topics).
# Claude and Qwen3 (262,144 tokens) see every passage whole.
for _name in ("gemma4b-umbrela", "llama8b-umbrela", "scout-umbrela"):
    OPEN_JUDGES[_name]["max_passage_chars"] = 280_000
JUDGES.update(OPEN_JUDGES)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def passage_for(judge, passage):
    """The passage as the judge sees it: cut to the judge's max_passage_chars (open-weight judges with a short
    context only), else whole. Returns (text, cut) with cut True when characters were removed."""
    cap = JUDGES[judge].get("max_passage_chars")
    text = passage.strip()
    return (text[:cap], True) if cap is not None and len(text) > cap else (text, False)


def openrouter_body(judge, query, passage):
    """Chat Completions body for one open-weight judge (OpenRouter docs: provider routing, parameters, plugins)."""
    j = JUDGES[judge]
    passage = passage_for(judge, passage)[0]
    return dict(model=j["model"], max_tokens=j["max_tokens"], temperature=j["temperature"], seed=j["seed"],
                messages=[{"role": "user", "content": render(j["prompt"], query, passage)}],
                provider=dict(order=[j["provider"]], allow_fallbacks=False, require_parameters=True,
                              quantizations=[j["quantization"]], data_collection="deny"),
                plugins=[{"id": "context-compression", "enabled": False}])

# $ per million tokens (input, output), list prices from the claude-api skill. Claude Haiku 5.5 bills a request
# whose prompt exceeds 100K tokens at the higher rate; the skill states no such tier for Claude Sonnet 5.5.
# The Batch API bills 50% of these.
LIST_PRICE = {HAIKU: (0.10, 0.50), SONNET: (2.0, 10.0)}
LONG_PRICE = {HAIKU: (100_000, (0.50, 2.50))}  # model: (prompt tokens above which, (input, output))


def render(prompt, query, passage):
    """Fill a prompt template. Plain replacement, so braces in passages are left alone."""
    tpl = template(prompt)
    if "{query}" not in tpl or "{passage}" not in tpl:
        raise ValueError(f"template {prompt} lacks a placeholder")
    return tpl.replace("{query}", query.strip()).replace("{passage}", passage.strip())


def request_params(judge, query, passage):
    """Messages API parameters for one Claude (judge, query, passage)."""
    j = JUDGES[judge]
    if j.get("api", "anthropic") != "anthropic":
        raise ValueError(f"{judge} is not a Claude judge; use openrouter_body")
    return dict(model=j["model"], max_tokens=j["max_tokens"], thinking=dict(j["thinking"]),
                output_config={"effort": j["effort"]},
                messages=[{"role": "user", "content": render(j["prompt"], query, passage)}])


_CID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def custom_id(judge, qid, docid):
    """Batch custom_id; '__' separates the parts (judge names and TREC ids never contain it)."""
    cid = f"{judge}__{qid}__{docid}"
    if not _CID.match(cid) or cid.count("__") != 2:
        raise ValueError(f"unusable custom_id: {cid!r}")
    return cid


def split_custom_id(cid):
    judge, qid, docid = cid.split("__")
    return judge, qid, docid


_FINAL = re.compile(r"final\s+score\s*(?:\(o\))?\s*[:=]?\s*\**\s*([0-3])(?![0-9.])", re.I)
_LAST = re.compile(r"^\W*(?:(?:final\s+)?(?:score|category|relevance(?:\s+category)?|answer|label)\s*[:=]?\s*)?\**\s*([0-3])\s*\**\.?\s*$", re.I)


def parse_label(text):
    """Integer label 0-3 from a model output, or None when it cannot be read unambiguously.

    Order: an explicit 'final score: X' (last one wins); otherwise the last non-empty line must be a bare label,
    optionally prefixed by score/category/relevance/answer/label and wrapped in markdown emphasis.
    """
    if not text:
        return None
    hits = _FINAL.findall(text)
    if hits:
        return int(hits[-1])
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    m = _LAST.match(lines[-1])
    return int(m.group(1)) if m else None


def ambiguous(text):
    """True when the output names more than one distinct label (as 'final score: X' or as a bare-label line).
    parse_label still returns the last one, as the prompts instruct; this only counts such outputs."""
    if not text:
        return False
    cands = {int(x) for x in _FINAL.findall(text)}
    cands |= {int(m.group(1)) for ln in text.splitlines() if (m := _LAST.match(ln.strip()))}
    return len(cands) > 1


def text_of(message_content):
    """Concatenated text blocks of a response's content (thinking blocks are skipped). Accepts SDK objects or dicts."""
    out = []
    for b in message_content:
        t = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
        if t == "text":
            out.append(b["text"] if isinstance(b, dict) else b.text)
    return "".join(out)


def cost_usd(model, input_tokens, output_tokens, batch=True):
    """Cost of ONE request (the long-prompt tier depends on the request's own prompt length)."""
    pin, pout = LIST_PRICE[model]
    if model in LONG_PRICE and input_tokens > LONG_PRICE[model][0]:
        pin, pout = LONG_PRICE[model][1]
    f = 0.5 if batch else 1.0
    return f * (input_tokens * pin + output_tokens * pout) / 1e6


def write_qrels(path, labels):
    """labels: {qid: {docid: int}} -> TREC qrels 'qid 0 docid label', sorted for stable diffs."""
    with open(path, "w") as fh:
        for q in sorted(labels):
            for d in sorted(labels[q]):
                fh.write(f"{q} 0 {d} {int(labels[q][d])}\n")


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def merge_label_dirs(dirs, read_qrels):
    """{judge: qrels} from the <judge>.txt files in several folders. A later folder only fills labels the
    earlier ones lack (used for retries), so a label once obtained is never replaced."""
    import os
    new = {}
    for folder in dirs:
        for fn in sorted(os.listdir(folder)):
            j = fn[:-4]
            if fn.endswith(".txt") and j in JUDGES:
                for t, docs in read_qrels(os.path.join(folder, fn)).items():
                    for d, v in docs.items():
                        new.setdefault(j, {}).setdefault(t, {}).setdefault(d, v)
    return new
