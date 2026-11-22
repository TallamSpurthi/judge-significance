"""Loading of qrels, judge label sets, runs and published reference values.

Expected layout of DATA_DIR (see the project README):

    qid_map.txt, docid_map.txt           LLMJudge ID maps (TREC id <tab> LLMJudge id)
    human_ids.txt                        LLMJudge human test qrels, TREC ids (25 topics, 4,423 pairs)
    judges/<name>.txt                    33 public judge label sets, LLMJudge ids
    runs/<run>                           35 TREC DL 2023 passage runs (or see below)
    dev_qrels_idx.txt                    optional: LLMJudge dev qrels, LLMJudge ids
    dl23/2023.qrels.pass.withDupes.txt   optional: full NIST DL-2023 passage qrels

The runs can live elsewhere, for example the official NIST files: pass runs_dir to load() or set the
environment variable LLMJUDGE_RUNS. Run files may be plain or gzipped; a run's name is its file name
without an "input." prefix and a ".gz" or ".txt" suffix, so NIST's input.<run>.gz and a bare <run> agree.
"""
import gzip
import os
from collections import defaultdict

FULL_QRELS = os.path.join("dl23", "2023.qrels.pass.withDupes.txt")
DEV_QRELS = "dev_qrels_idx.txt"


def read_map(path):
    """Map LLMJudge ids to TREC ids."""
    m = {}
    with open(path) as fh:
        for line in fh:
            trec_id, llmjudge_id = line.split()
            m[llmjudge_id] = trec_id
    return m


def read_qrels(path, qmap=None, dmap=None):
    """Raw integer labels as written in the file (no clipping). With qmap/dmap, ids are mapped to TREC ids."""
    q = defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) < 4:
                continue
            qid, did, rel = f[0], f[2], int(float(f[3]))
            if qmap is not None:
                qid, did = qmap[qid], dmap[did]
            q[qid][did] = rel
    return dict(q)


def relabel(qrels, mode):
    """Out-of-range label handling.

    raw     keep labels as submitted (trec_eval uses the value as the gain)
    clip03  clip into [0, 3]
    oob0    any label outside {0, 1, 2, 3} becomes 0
    """
    if mode == "raw":
        return qrels
    out = {}
    for q, d in qrels.items():
        if mode == "clip03":
            out[q] = {k: min(max(v, 0), 3) for k, v in d.items()}
        elif mode == "oob0":
            out[q] = {k: (v if 0 <= v <= 3 else 0) for k, v in d.items()}
        else:
            raise ValueError(mode)
    return out


def run_name(filename):
    """input.<run>.gz, input.<run>.txt or <run> -> <run>."""
    name = filename[len("input."):] if filename.startswith("input.") else filename
    for ext in (".gz", ".txt"):
        if name.endswith(ext):
            name = name[:-len(ext)]
    return name


def read_runs_dir(d, topics=None):
    """{run name: run} for every file in directory d."""
    runs = {}
    for fn in sorted(os.listdir(d)):
        name = run_name(fn)
        if name in runs:
            raise ValueError(f"two files in {d} give the run name {name!r}")
        runs[name] = read_run(os.path.join(d, fn), topics)
    return runs


def read_run(path, topics=None):
    """TREC run as {qid: {docid: score}}; keeps the first occurrence of a duplicated docid. Reads .gz files."""
    r = defaultdict(dict)
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
        for line in fh:
            f = line.split()
            if topics is not None and f[0] not in topics:
                continue
            r[f[0]].setdefault(f[2], float(f[4]))
    return dict(r)


def load(data_dir, runs_dir=None):
    """Load everything in DATA_DIR. Runs are restricted to topics that have qrels. Runs are read from
    runs_dir, else $LLMJUDGE_RUNS, else DATA_DIR/runs."""
    qmap = read_map(os.path.join(data_dir, "qid_map.txt"))
    dmap = read_map(os.path.join(data_dir, "docid_map.txt"))
    human = read_qrels(os.path.join(data_dir, "human_ids.txt"))
    judges = {}
    for fn in sorted(os.listdir(os.path.join(data_dir, "judges"))):
        judges[fn[:-4]] = read_qrels(os.path.join(data_dir, "judges", fn), qmap, dmap)
    full_path = os.path.join(data_dir, FULL_QRELS)
    full = read_qrels(full_path) if os.path.exists(full_path) else None
    keep = set(full) if full else set(human)
    runs = read_runs_dir(runs_dir or os.environ.get("LLMJUDGE_RUNS") or os.path.join(data_dir, "runs"), keep)
    return dict(human=human, judges=judges, full=full, runs=runs,
                test_topics=sorted(human), qmap=qmap, dmap=dmap)


def load_dev_qrels(data_dir, data):
    """LLMJudge dev qrels mapped to TREC ids, or None if the file is absent."""
    path = os.path.join(data_dir, DEV_QRELS)
    if not os.path.exists(path):
        return None
    return read_qrels(path, data["qmap"], data["dmap"])


def read_published(path):
    """Published Table 3 values: {submission: {kappa, alpha, tau, rho}}."""
    pub = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("submission"):
                continue
            name, kappa, alpha, tau, rho = line.rstrip("\n").split("\t")
            pub[name] = dict(kappa=float(kappa), alpha=float(alpha), tau=float(tau), rho=float(rho))
    return pub
