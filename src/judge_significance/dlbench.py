"""Loader for the TREC DL 2019 and 2020 passage benchmark of LLM relevance judges.

Two public label collections are combined, each judging every pair in the NIST passage qrels:

    Arabzadeh and Clarke (SIGIR 2025), github.com/Narabzad/llm-relevance-judgement-comparison
        GPT-4o: binary, UMBRELA, EXAM binary, EXAM graded max, EXAM graded mean
        Llama 3.2: binary, UMBRELA (4 to 6 repeated samples per pair)
    Farzi and Dietz (SIGIR 2025), github.com/TREMA-UNH/appendix-umbrella-other-llm
        UMBRELA zero-shot Bing prompt with DeepSeek-V3, Llama-3.3-70B, Llama-3-8B, Flan-T5-large;
        zero-shot basic prompt with Llama-3-8B

Expected layout of DATA_DIR (both repositories cloned with a sparse checkout, paths as in the repositories):

    arabzadeh/data/qrels.{dl19,dl20}-passage.txt       NIST passage qrels (43 and 54 topics)
    arabzadeh/raw qrels/<model>/<method>/*_{dl19,dl20}.txt
    arabzadeh/runs/{dl19,dl20}/input.<run>.txt          37 and 59 TREC passage runs (copies; see below)
    arabzadeh/runs/{dl19,dl20}.ndcg_cut10              per-topic human nDCG@10 (CSV runid,topic,ndcg_cut_10)
    farzi/result logs/{dl19,dl20}_test_zeroshot_*.jsonl

The runs can be read from elsewhere, for example the official NIST files: pass runs_dir to load_year()
or set DLBENCH_RUNS to a folder holding dl19/ and dl20/ subfolders of run files (plain or gzipped,
named input.<run>.gz, input.<run>.txt or <run>). The per-topic human nDCG@10 is always read from
arabzadeh/runs/{dl19,dl20}.ndcg_cut10.

Label rules (see LABEL_RULES): judges whose labels are integers are used as distributed (no clipping;
trec_eval uses the label as the gain). Repeated samples of one pair are aggregated by AGGREGATORS.
Fractional labels cannot be passed to pytrec_eval; they are binned to 0-3 only for the sensitivity check.
A pair a judge did not label is left out of its qrels, so trec_eval treats it as non-relevant.
"""
import json
import math
import os
import re
from collections import Counter, defaultdict

from .data import read_qrels, read_run, read_runs_dir, run_name

YEARS = ("dl19", "dl20")
ARAB = "arabzadeh"
FARZI = "farzi"

# (config name, source, relative path with {y} for the year, scale)
# scale: "int" integer labels used as-is; "frac:<max>" fractional labels on [0, max].
ARAB_CONFIGS = (
    ("gpt-4o/binary", "raw qrels/gpt-4o/binary/binary_{y}.txt", "int"),
    ("gpt-4o/umbrela", "raw qrels/gpt-4o/umbrela/umbrela_{y}.txt", "int"),
    ("gpt-4o/exam-graded-max", "raw qrels/gpt-4o/exam graded max/exam_graded_max_{y}.txt", "int"),
    ("gpt-4o/exam-graded-mean", "raw qrels/gpt-4o/exam graded mean/exam_graded_mean_{y}.txt", "frac:5"),
    ("gpt-4o/exam-binary", "raw qrels/gpt-4o/exam binary/exam_binary_{y}.txt", "frac:1"),
    ("llama3.2/binary", "raw qrels/llama3.2/binary/binary_llama3.2_{y}.txt", "int"),
    ("llama3.2/umbrela", "raw qrels/llama3.2/umbrela/umbrella_llama3.2_{y}.txt", "int"),
)
FARZI_CONFIGS = (
    ("deepseek-v3/umbrela", "result logs/{y}_test_zeroshot_bing_DSV3.jsonl"),
    ("llama-3.3-70b/umbrela", "result logs/{y}_test_zeroshot_bing_Llama-3.3-70B.jsonl"),
    ("llama-3-8b/umbrela", "result logs/{y}_test_zeroshot_bing_Llama-3-8B-Instruct.jsonl"),
    ("flan-t5-large/umbrela", "result logs/{y}_test_zeroshot_bing_flan-t5-large.jsonl"),
    ("llama-3-8b/umbrela-basic", "result logs/{y}_test_zeroshot_basic_Llama-3-8B-Instruct.jsonl"),
)
# Farzi and Dietz, Table 2, Kendall tau on nDCG@10 (UMBRELA zero-shot Bing prompt)
FD_TABLE2_TAU = {
    "dl19": {"deepseek-v3/umbrela": 0.898, "llama-3.3-70b/umbrela": 0.869,
             "llama-3-8b/umbrela": 0.894, "flan-t5-large/umbrela": 0.862},
    "dl20": {"deepseek-v3/umbrela": 0.940, "llama-3.3-70b/umbrela": 0.911,
             "llama-3-8b/umbrela": 0.870, "flan-t5-large/umbrela": 0.717},
}
# Farzi and Dietz, Table 2, Cohen's kappa against NIST labels: (four-scale, 01-vs-23)
FD_TABLE2_KAPPA = {
    "dl19": {"deepseek-v3/umbrela": (0.360, 0.518), "llama-3.3-70b/umbrela": (0.266, 0.447),
             "llama-3-8b/umbrela": (0.108, 0.244), "flan-t5-large/umbrela": (0.050, 0.292)},
    "dl20": {"deepseek-v3/umbrela": (0.358, 0.426), "llama-3.3-70b/umbrela": (0.247, 0.344),
             "llama-3-8b/umbrela": (0.115, 0.187), "flan-t5-large/umbrela": (0.036, 0.231)},
}
FD_SYSTEMS = {"dl19": 36, "dl20": 59}

LABEL_RULES = """\
integer labels: used as distributed, no clipping (GPT-4o EXAM graded max is on 0-5, the rest on 0-1 or 0-3)
repeated samples of a pair (Llama 3.2 UMBRELA): mean of all samples, rounded half up (primary)
duplicated lines in a Farzi-Dietz log: the last line wins (primary)
Farzi-Dietz labels: the stored final_relevance_score (primary); re-parsed from LLMs_output (sensitivity only)
fractional labels (EXAM binary on [0, 1], EXAM graded mean on [0, 5]): excluded from the primary set;
  sensitivity only: floor(3 * x / max + 0.5), i.e. scaled to 0-3 and rounded half up
pairs a judge did not label: absent from its qrels (trec_eval default: unjudged = non-relevant)"""


def round_half_up(x):
    return int(math.floor(x + 0.5))


def agg_mean(vals):
    """Mean of the samples, rounded half up."""
    return round_half_up(sum(vals) / len(vals))


def agg_mode(vals):
    """Most frequent sample; ties go to the lowest label."""
    c = Counter(vals)
    top = max(c.values())
    return min(v for v, n in c.items() if n == top)


def agg_first(vals):
    return vals[0]


def agg_last(vals):
    return vals[-1]


def agg_mean_first4(vals):
    """Mean of the first four samples (every pair has at least four), rounded half up."""
    return agg_mean(vals[:4])


AGGREGATORS = {"mean": agg_mean, "mode": agg_mode, "first": agg_first, "last": agg_last,
               "mean4": agg_mean_first4}


def bin_fraction(x, scale_max):
    """Fractional label on [0, scale_max] to 0-3: floor(3 x / scale_max + 0.5)."""
    return min(3, max(0, round_half_up(3 * x / scale_max)))


def read_label_lines(path):
    """Text qrels as {qid: {docid: [label, ...]}} in file order; labels as floats."""
    out = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) < 4:
                continue
            out[f[0]][f[2]].append(float(f[3]))
    return {q: dict(d) for q, d in out.items()}


def reparse_last_label(text):
    """Last standalone 0-3 digit in an LLM output (0 if none).

    The stored final_relevance_score in the Farzi-Dietz logs falls back to the first standalone 0-3 digit
    when an output lacks '##final score: X' and 'O: X', so '## Step 1: M = 2 ... ## Step 3: O = 2' is
    stored as 1. This re-parse is used only as a sensitivity check."""
    m = re.findall(r"\b[0-3]\b", text)
    return int(m[-1]) if m else 0


def read_farzi_log(path, reparse=False):
    """Farzi-Dietz result log (JSONL with qidx, docidx, final_relevance_score) as {qid: {docid: [label, ...]}}.
    reparse=True takes reparse_last_label(LLMs_output) instead of the stored score."""
    out = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            lab = reparse_last_label(r["LLMs_output"]) if reparse else r["final_relevance_score"]
            out[str(r["qidx"])][str(r["docidx"])].append(float(lab))
    return {q: dict(d) for q, d in out.items()}


def sample_stats(samples):
    """Counts of pairs by number of samples, and pairs whose samples disagree."""
    mult = Counter(len(v) for d in samples.values() for v in d.values())
    conflict = sum(len(set(v)) > 1 for d in samples.values() for v in d.values())
    return dict(sorted(mult.items())), conflict


def is_integer_valued(samples):
    return all(float(x).is_integer() for d in samples.values() for v in d.values() for x in v)


def to_qrels(samples, agg="mean", frac_max=None):
    """Collapse samples to one integer label per pair. With frac_max, labels are binned with bin_fraction
    after aggregation by the plain mean (agg is then ignored)."""
    out = {}
    for q, d in samples.items():
        if frac_max is not None:
            out[q] = {doc: bin_fraction(sum(v) / len(v), frac_max) for doc, v in d.items()}
        else:
            f = AGGREGATORS[agg]
            out[q] = {doc: int(f([int(x) for x in v])) for doc, v in d.items()}
    return out


def coverage(qrels, human):
    """(pairs of human that qrels labels, pairs of human, pairs in qrels that human does not judge)."""
    hit = sum(1 for q, d in human.items() for doc in d if doc in qrels.get(q, {}))
    total = sum(len(d) for d in human.values())
    extra = sum(1 for q, d in qrels.items() for doc in d if doc not in human.get(q, {}))
    return hit, total, extra


def restrict(qrels, human):
    """Keep only pairs that the human qrels judge."""
    return {q: {doc: v for doc, v in d.items() if doc in human.get(q, {})} for q, d in qrels.items() if q in human}


def same_labels(a, b):
    """True if two qrels have identical pairs and labels."""
    return a.keys() == b.keys() and all(a[q] == b[q] for q in a)


def read_ndcg_csv(path):
    """Per-topic trec_eval values (CSV runid,topic,value; 'average' rows skipped) as {run: {topic: value}}."""
    out = defaultdict(dict)
    with open(path) as fh:
        next(fh)
        for line in fh:
            run, topic, val = line.strip().split(",")
            if topic != "average":
                out[run][topic] = float(val)
    return dict(out)


def load_runs(data_dir, year, topics, runs_dir=None):
    """Runs of one year from runs_dir/<year>, else $DLBENCH_RUNS/<year>, else DATA_DIR/arabzadeh/runs/<year>."""
    base = runs_dir or os.environ.get("DLBENCH_RUNS") or os.path.join(data_dir, ARAB, "runs")
    return read_runs_dir(os.path.join(base, year), set(topics))


def load_judge_samples(data_dir, year):
    """{config: (samples, scale, source path)} for every configuration whose file exists and is non-empty."""
    out = {}
    for name, rel, scale in ARAB_CONFIGS:
        p = os.path.join(data_dir, ARAB, rel.format(y=year))
        if os.path.exists(p) and os.path.getsize(p) > 0:
            out[name] = (read_label_lines(p), scale, p)
    for name, rel in FARZI_CONFIGS:
        p = os.path.join(data_dir, FARZI, rel.format(y=year))
        if os.path.exists(p) and os.path.getsize(p) > 0:
            out[name] = (read_farzi_log(p), "int", p)
    return out


def default_agg(name):
    """Primary aggregation rule: mean for repeated samples (Llama 3.2 UMBRELA), last line for Farzi-Dietz logs."""
    return "last" if name in dict((n, r) for n, r in FARZI_CONFIGS) else "mean"


def build_judges(samples_by_config, human, agg_overrides=None, include_fractional=False, min_coverage=0.999):
    """Primary judge qrels and a per-config report.

    Returns (judges, report): judges = {config: qrels restricted to human pairs}; report rows are dicts
    with name, scale, coverage, multiplicity, conflicts, included and reason."""
    agg_overrides = agg_overrides or {}
    judges, report = {}, []
    raw_seen = {}
    for name, (samples, scale, path) in samples_by_config.items():
        mult, conflict = sample_stats(samples)
        row = dict(name=name, scale=scale, path=path, mult=mult, conflict=conflict, note="")
        raw = {q: {doc: tuple(v) for doc, v in d.items()} for q, d in samples.items()}
        dup = [other for other, r in raw_seen.items() if same_labels(r, raw)]
        raw_seen[name] = raw
        if dup:
            row["note"] = f"labels identical to {dup[0]}"
        if scale.startswith("frac"):
            if is_integer_valued(samples):
                row["note"] = (row["note"] + "; " if row["note"] else "") + "all labels integer-valued"
            qr = to_qrels(samples, frac_max=float(scale.split(":")[1])) if include_fractional else None
        else:
            qr = to_qrels(samples, agg=agg_overrides.get(name, default_agg(name)))
        hit, total, extra = coverage(qr if qr is not None else samples, human)
        row.update(covered=hit, total=total, extra=extra)
        if qr is None:
            row.update(included=False, reason="fractional labels")
        elif dup:
            row.update(included=False, reason=f"duplicate of {dup[0]}")
        elif hit / total < min_coverage:
            row.update(included=False, reason=f"coverage {hit / total:.4f} < {min_coverage}")
        else:
            row.update(included=True, reason="")
            judges[name] = restrict(qr, human)
        report.append(row)
    return judges, report


def load_year(data_dir, year, read_runs=True, runs_dir=None):
    """Human qrels, topics, runs, raw judge samples and the published per-topic human nDCG@10 for one year."""
    human = read_qrels(os.path.join(data_dir, ARAB, "data", f"qrels.{year}-passage.txt"))
    topics = sorted(human)
    runs = load_runs(data_dir, year, topics, runs_dir) if read_runs else None
    ndcg = read_ndcg_csv(os.path.join(data_dir, ARAB, "runs", f"{year}.ndcg_cut10"))
    return dict(year=year, human=human, topics=topics, runs=runs, samples=load_judge_samples(data_dir, year),
                published_ndcg=ndcg)
