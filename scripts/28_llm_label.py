"""Step 28. Label TREC DL 2023 query-passage pairs with Claude judges (pilot and main study).

Subcommands (stages and gates as listed below; the released labels, model identifiers and settings are
described in results/llm-judges/README.md):
  selfcheck DATA_DIR                      Gate 0: every published LLMJudge judge, written by this script's qrels
                                          writer and read back, reproduces its published tau (no API).
  pairs-pilot DATA_DIR SRC_DIR --out F    The 4,423 LLMJudge test pairs with query and passage text and the
                                          human label (SRC_DIR: the LLMJudge repository's data/ folder).
  ids DATA_DIR --out F                    Passage ids of all judged DL 2023 pairs (input to step 27).
  pairs-main DATA_DIR --passages P --queries Q --out F
                                          All judged DL 2023 pairs with text (P from step 27, Q = 2023_queries.tsv).
  sync PAIRS --out DIR [--n 20] [--max-usd 0.10]
                                          Stage 1: synchronous calls for n pairs (stratified by human label) and
                                          every judge; stops when the spend at list price reaches --max-usd.
  submit PAIRS --calib DIR --out DIR --max-usd X [--judges ...] [--calib-pairs F ...]
                                          Stage 2/3: one Message Batch per judge. Refuses to submit unless the
                                          cost projected from measured token counts (a linear model in query+passage
                                          length, fitted per judge on `sync` or fetched batch outputs in DIR) is
                                          <= X, or if a batch for that judge already exists in DIR.
  run PAIRS --out DIR --max-usd X --judges ... [--workers 8] [--retry-failed]
                                          Synchronous, parallel and resumable alternative to submit/fetch, used
                                          when batches stay queued (list price, twice the batch price).
  fetch DIR                               Status of the batches in DIR; when a batch has ended, writes raw
                                          outputs, parse failures and a qrels file per judge, and the actual cost.
  evaluate DATA_DIR DIR [DIR ...]         Gate 2: tau and kappa of each new judge on the 25 LLMJudge topics,
                                          label distributions and disagreement between the new judges.

The API key is read by the SDK from ANTHROPIC_API_KEY and never printed or written. The judge haiku-llmjudge also
needs LLMJUDGE_PROMPT, the path of the LLMJudge repository's sample-prompt.txt, which is not stored here.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict

import numpy as np

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance import llm_label as L
from judge_significance.simulation import stat_rows

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PUBLISHED = os.path.join(ROOT, "resources", "llmjudge_table3.tsv")
_spec = importlib.util.spec_from_file_location("cov15", os.path.join(HERE, "15_real_judge_coverage.py"))
_cov15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cov15)


# ---------------------------------------------------------------- tau on the 25 LLMJudge topics (as in the paper)
def tau_on_test(data, qrels):
    runs, test = data["runs"], data["test_topics"]
    systems = _cov15.table3_systems(runs)
    H = E.score_matrix(runs, systems, data["human"], test, "ndcg_cut_10")
    J = E.score_matrix(runs, systems, qrels, test, "ndcg_cut_10")
    return float(stat_rows(H.mean(1)[None, :], J.mean(1)[None, :])[0])


def kappa(human, qrels):
    """Unweighted Cohen's kappa over the human-judged pairs (labels clipped to 0-3); missing labels count as 0."""
    C = np.zeros((4, 4))
    for q, docs in human.items():
        for d, h in docs.items():
            j = qrels.get(q, {}).get(d, 0)
            C[min(max(int(h), 0), 3), min(max(int(j), 0), 3)] += 1
    n = C.sum()
    po, pe = np.trace(C) / n, (C.sum(0) * C.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe))


def cmd_selfcheck(args):
    data = D.load(args.data_dir)
    pub = D.read_published(PUBLISHED)
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name in sorted(data["judges"]):
            path = os.path.join(tmp, name + ".txt")
            L.write_qrels(path, data["judges"][name])
            back = D.read_qrels(path)
            same = back == data["judges"][name]
            t = tau_on_test(data, back)
            ok = same and round(t, 4) == pub[name]["tau"]
            bad += not ok
            if not ok:
                print(f"  MISMATCH {name}: roundtrip identical={same}, tau {t:.4f} vs published {pub[name]['tau']}")
    print(f"selfcheck: {len(data['judges']) - bad}/{len(data['judges'])} judges reproduce their published tau "
          f"through write_qrels/read_qrels")
    sys.exit(1 if bad else 0)


# ---------------------------------------------------------------- pair lists
def cmd_pairs_pilot(args):
    data = D.load(args.data_dir)
    qmap, dmap, human = data["qmap"], data["dmap"], data["human"]
    queries = dict(ln.rstrip("\n").split("\t", 1) for ln in open(os.path.join(args.src_dir, "llm4eval_query_2024.txt")))
    docs = {}
    with open(os.path.join(args.src_dir, "llm4eval_document_2024.jsonl")) as fh:
        for ln in fh:
            r = json.loads(ln)
            docs[r["docid"]] = r["doc"]
    rows, missing = [], 0
    with open(os.path.join(args.src_dir, "llm4eval_test_qrel_2024.txt")) as fh:
        for ln in fh:
            qi, _, di = ln.split()[:3]
            if qi not in queries or di not in docs:
                missing += 1
                continue
            q, d = qmap[qi], dmap[di]
            rows.append(dict(qid=q, docid=d, query=queries[qi], passage=docs[di], human=human.get(q, {}).get(d)))
    L.write_jsonl(args.out, rows)
    print(f"wrote {len(rows):,} pairs on {len({r['qid'] for r in rows})} topics to {os.path.basename(args.out)} "
          f"({missing} without text; {sum(r['human'] is None for r in rows)} without a human label)")
    sys.exit(1 if missing else 0)


def _full_pairs(data_dir):
    full = D.read_qrels(os.path.join(data_dir, D.FULL_QRELS))
    return full


def cmd_ids(args):
    full = _full_pairs(args.data_dir)
    ids = sorted({d for docs in full.values() for d in docs})
    with open(args.out, "w") as fh:
        fh.write("\n".join(ids) + "\n")
    print(f"{sum(len(v) for v in full.values()):,} judged pairs on {len(full)} topics; {len(ids):,} distinct passages")


def cmd_pairs_main(args):
    full = _full_pairs(args.data_dir)
    passages = {r["pid"]: r["passage"] for r in L.read_jsonl(args.passages)}
    queries = dict(ln.rstrip("\n").split("\t", 1) for ln in open(args.queries))
    rows, miss_q, miss_p = [], set(), set()
    for q in sorted(full):
        for d in sorted(full[q]):
            if q not in queries:
                miss_q.add(q)
            elif d not in passages:
                miss_p.add(d)
            else:
                rows.append(dict(qid=q, docid=d, query=queries[q], passage=passages[d], human=full[q][d]))
    L.write_jsonl(args.out, rows)
    print(f"wrote {len(rows):,} pairs; missing queries {len(miss_q)}, missing passages {len(miss_p)}")
    sys.exit(1 if miss_q or miss_p else 0)


# ---------------------------------------------------------------- API calls
def _client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set in this shell; set it in your own terminal first.")
    import anthropic
    return anthropic.Anthropic()


def _claude_call(client, judge, model, query, passage):
    """One synchronous Messages API call for a Claude judge; returns the record fields `run` stores."""
    try:
        msg = client.messages.create(**L.request_params(judge, query, passage))
        text = L.text_of(msg.content)
        u = msg.usage
        return dict(type="succeeded", text=text, label=L.parse_label(text), stop_reason=msg.stop_reason,
                    ambiguous=L.ambiguous(text), input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                    cost_usd=L.cost_usd(model, u.input_tokens, u.output_tokens, batch=False), message_id=msg.id,
                    model=msg.model)
    except Exception as e:  # recorded, not fatal; --retry-failed asks again
        return dict(type="errored", label=None, error=f"{type(e).__name__}: {e}"[:500], cost_usd=0.0)


def _openrouter_call(judge, query, passage, tries=4, timeout=600):
    """One OpenRouter Chat Completions call for an open-weight judge; returns the record fields `run` stores.
    Retries 408/429/5xx/529 and network errors with backoff; other HTTP errors (for example 400 context overflow,
    404/503 when the pinned endpoint is unavailable) are returned as errors without retry. The key comes from
    OPENROUTER_API_KEY and is sent only in the Authorization header; it is never logged."""
    import time
    import urllib.error
    import urllib.request
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set in this shell.")
    body = json.dumps(L.openrouter_body(judge, query, passage)).encode()
    for k in range(tries):
        req = urllib.request.Request(L.OPENROUTER_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            if "error" in d:
                return dict(type="errored", label=None, error=json.dumps(d["error"])[:500], cost_usd=0.0)
            ch, u = d["choices"][0], d.get("usage") or {}
            text = (ch.get("message") or {}).get("content") or ""
            return dict(type="succeeded", text=text, label=L.parse_label(text), stop_reason=ch.get("finish_reason"),
                        native_finish_reason=ch.get("native_finish_reason"), ambiguous=L.ambiguous(text),
                        input_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"),
                        cost_usd=float(u.get("cost") or 0.0), message_id=d.get("id"), model=d.get("model"),
                        provider=d.get("provider"), passage_cut=L.passage_for(judge, passage)[1])
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:500]
            if e.code not in (408, 429, 500, 502, 503, 504, 529) or k == tries - 1:
                return dict(type="errored", label=None, error=f"HTTP {e.code}: {msg}", cost_usd=0.0)
        except (OSError, ValueError) as e:  # URLError, timeouts, SSL and connection errors, malformed JSON
            if k == tries - 1:
                return dict(type="errored", label=None, error=f"{type(e).__name__}: {e}"[:500], cost_usd=0.0)
        time.sleep(2 ** k * 5)


def _dump(obj):
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


def _claude_only(judges, cmd):
    bad = [j for j in judges if L.JUDGES[j].get("api", "anthropic") != "anthropic"]
    if bad:
        sys.exit(f"{cmd} is for Claude judges only; use `run` for {', '.join(bad)}")


def cmd_sync(args):
    _claude_only(args.judges, "sync")
    pairs = L.read_jsonl(args.pairs)
    rng = np.random.default_rng(args.seed)
    by_label = defaultdict(list)
    for r in pairs:
        by_label[r.get("human")].append(r)
    pick, k = [], 0
    keys = sorted(by_label, key=lambda x: (x is None, x))
    while len(pick) < args.n and any(by_label.values()):  # round-robin over human labels
        lab = keys[k % len(keys)]
        k += 1
        if by_label[lab]:
            pick.append(by_label[lab].pop(int(rng.integers(len(by_label[lab])))))
    client = _client()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "sync_raw.jsonl")
    spent, rows = 0.0, []
    with open(out_path, "w") as fh:
        for judge in args.judges:
            for r in pick:
                if spent >= args.max_usd:
                    print(f"stopped: spend ${spent:.4f} reached --max-usd {args.max_usd}")
                    break
                params = L.request_params(judge, r["query"], r["passage"])
                rec = dict(judge=judge, qid=r["qid"], docid=r["docid"], human=r.get("human"))
                try:
                    msg = client.messages.create(**params)
                    text = L.text_of(msg.content)
                    u = msg.usage
                    c = L.cost_usd(params["model"], u.input_tokens, u.output_tokens, batch=False)
                    spent += c
                    rec.update(ok=True, stop_reason=msg.stop_reason, text=text, label=L.parse_label(text),
                               input_tokens=u.input_tokens, output_tokens=u.output_tokens, cost_usd=c,
                               block_types=[b.type for b in msg.content], response=_dump(msg))
                except Exception as e:  # recorded, not fatal: the gate counts these
                    rec.update(ok=False, error=f"{type(e).__name__}: {e}")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows.append(rec)
    print(f"sync: {len(rows)} calls, spend ${spent:.4f} at list price; raw outputs in {out_path}")
    for judge in args.judges:
        jr = [r for r in rows if r["judge"] == judge]
        ok = [r for r in jr if r.get("ok")]
        parsed = [r for r in ok if r["label"] is not None]
        print(f"  {judge}: {len(jr)} calls, {len(ok)} returned, {len(parsed)} parsed; "
              f"stop {dict(Counter(r['stop_reason'] for r in ok))}; "
              f"mean tokens in {np.mean([r['input_tokens'] for r in ok]) if ok else float('nan'):.0f} "
              f"out {np.mean([r['output_tokens'] for r in ok]) if ok else float('nan'):.1f}; "
              f"agree with human {sum(r['label'] == r['human'] for r in parsed)}/{len(parsed)}")
        for r in jr:
            if not r.get("ok") or r["label"] is None:
                print(f"    FAILED {r['qid']} {r['docid']}: {r.get('error') or repr(r.get('text'))[:200]}")


def _latest(path):
    """Latest record per custom_id in a `run` raw file (later lines replace earlier ones)."""
    last = {}
    if os.path.exists(path):
        for r in L.read_jsonl(path):
            last[r["custom_id"]] = r
    return last


def _write_outputs(out_dir, judge, last):
    labels, failed = defaultdict(dict), []
    for r in last.values():
        _, q, d = L.split_custom_id(r["custom_id"])
        if r.get("label") is None:
            failed.append(r)
        else:
            labels[q][d] = r["label"]
    L.write_qrels(os.path.join(out_dir, f"{judge}.txt"), labels)
    L.write_jsonl(os.path.join(out_dir, f"{judge}_failed.jsonl"), failed)
    return labels, failed


def cmd_run(args):
    """Synchronous, parallel, resumable labelling (used when batches stay queued). Writes the same files as
    `fetch` (<judge>_raw.jsonl, <judge>.txt, <judge>_failed.jsonl) plus each response id and cost. A rerun skips
    pairs already answered; with --retry-failed it re-asks, once, the pairs whose answer did not parse."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    pairs = L.read_jsonl(args.pairs)
    apis = {L.JUDGES[j].get("api", "anthropic") for j in args.judges}
    client = _client() if "anthropic" in apis else None
    if "openrouter" in apis and not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set in this shell.")
    os.makedirs(args.out, exist_ok=True)
    lock, state = threading.Lock(), dict(spent=0.0, stop=False)
    for judge in args.judges:  # spend already recorded in the folder counts against --max-usd
        state["spent"] += sum(r.get("cost_usd", 0.0) for r in L.read_jsonl(os.path.join(args.out, f"{judge}_raw.jsonl"))) \
            if os.path.exists(os.path.join(args.out, f"{judge}_raw.jsonl")) else 0.0
    print(f"already spent in {args.out}: ${state['spent']:.4f}; limit --max-usd {args.max_usd}")
    for judge in args.judges:
        raw = os.path.join(args.out, f"{judge}_raw.jsonl")
        last = _latest(raw)
        todo = []
        for r in pairs:
            cid = L.custom_id(judge, r["qid"], r["docid"])
            prev = last.get(cid)
            if prev is None or (args.retry_failed and prev.get("label") is None and prev.get("attempt", 1) < 2):
                todo.append((r, cid, 1 if prev is None else prev.get("attempt", 1) + 1))
        print(f"{judge}: {len(todo):,} to ask ({len(last):,} already in {raw})", flush=True)
        model = L.JUDGES[judge]["model"]
        done = [0]

        def ask(item, judge=judge, model=model, done=done):
            r, cid, attempt = item
            with lock:
                if state["stop"] or state["spent"] >= args.max_usd:
                    state["stop"] = True
                    return None
            rec = dict(custom_id=cid, attempt=attempt)
            try:
                if L.JUDGES[judge].get("api") == "openrouter":
                    rec.update(_openrouter_call(judge, r["query"], r["passage"]))
                else:
                    rec.update(_claude_call(client, judge, model, r["query"], r["passage"]))
            except Exception as e:  # never let one request stop the run; recorded and retried with --retry-failed
                rec.update(type="errored", label=None, error=f"{type(e).__name__}: {e}"[:500], cost_usd=0.0)
            with lock:
                state["spent"] += rec["cost_usd"]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[0] += 1
                if done[0] % 250 == 0:
                    print(f"  {judge}: {done[0]:,}/{len(todo):,}; spent ${state['spent']:.3f}", flush=True)
            return rec

        with open(raw, "a") as fh, ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(ask, todo))
        last = _latest(raw)
        labels, failed = _write_outputs(args.out, judge, last)
        ok = [r for r in last.values() if r["type"] == "succeeded"]
        print(f"  {judge}: {len(last):,} answered; labelled {sum(len(v) for v in labels.values()):,}; failed {len(failed)} "
              f"({len(failed) / max(len(last), 1):.2%}); two labels in one output {sum(r['ambiguous'] for r in ok)}; "
              f"stop reasons {dict(Counter(r['stop_reason'] for r in ok))}; errors "
              f"{dict(Counter(r['error'].split(':')[0] for r in last.values() if r['type'] == 'errored'))}", flush=True)
        if state["stop"]:
            print(f"stopped: spend ${state['spent']:.4f} reached --max-usd {args.max_usd}; rerun to resume")
            break
    print(f"total spent in {args.out} (list price): ${state['spent']:.4f}")


def _calibration(calib_dir, chars_of):
    """Per judge: input tokens as a linear function of query+passage characters (tokens = a + b * chars), the
    mean output tokens, and the number of calls, from `sync` outputs (sync_raw.jsonl) or fetched batches
    (<judge>_raw.jsonl) in calib_dir. chars_of maps (qid, docid) to query+passage characters.

    Nov 9 pilot: a projection from mean tokens alone underestimated the Sonnet batch by 21%, because the
    calibration pairs had shorter passages (460 characters on average) than the set (957)."""
    rows = []
    path = os.path.join(calib_dir, "sync_raw.jsonl")
    if os.path.exists(path):
        rows += [dict(judge=r["judge"], qid=r["qid"], docid=r["docid"], tin=r["input_tokens"], tout=r["output_tokens"])
                 for r in L.read_jsonl(path) if r.get("ok")]
    for fn in sorted(os.listdir(calib_dir)):
        if fn.endswith("_raw.jsonl") and fn[:-10] in L.JUDGES:
            for r in L.read_jsonl(os.path.join(calib_dir, fn)):
                if r.get("type") == "succeeded":
                    j, q, d = L.split_custom_id(r["custom_id"])
                    rows.append(dict(judge=j, qid=q, docid=d, tin=r["input_tokens"], tout=r["output_tokens"]))
    cal = {}
    for judge in sorted({r["judge"] for r in rows}):
        jr = [r for r in rows if r["judge"] == judge and (r["qid"], r["docid"]) in chars_of]
        x = np.array([chars_of[(r["qid"], r["docid"])] for r in jr], float)
        y = np.array([r["tin"] for r in jr], float)
        if len(jr) < 10 or np.ptp(x) < 200:
            raise SystemExit(f"calibration for {judge}: {len(jr)} usable calls; need at least 10 with varied lengths")
        b, a = np.polyfit(x, y, 1)
        cal[judge] = dict(a=float(a), b=float(b), tout=float(np.mean([r["tout"] for r in jr])), n=len(jr))
    return cal


def cmd_submit(args):
    _claude_only(args.judges, "submit")
    pairs = L.read_jsonl(args.pairs)
    chars = np.array([len(r["query"]) + len(r["passage"]) for r in pairs], float)
    calib_chars = {}
    for f in args.calib_pairs or [args.pairs]:
        for r in L.read_jsonl(f):
            calib_chars[(r["qid"], r["docid"])] = len(r["query"]) + len(r["passage"])
    cal = _calibration(args.calib, calib_chars)
    os.makedirs(args.out, exist_ok=True)
    reg_path = os.path.join(args.out, "batches.json")
    reg = json.load(open(reg_path)) if os.path.exists(reg_path) else {}
    proj, total, complete = {}, 0.0, set()
    for judge in args.judges:
        if judge not in cal:
            sys.exit(f"no calibration for {judge} in {args.calib}")
        if judge in reg and sum(p["n_requests"] for p in reg[judge].get("parts", [reg[judge]])) >= len(pairs):
            print(f"  {judge}: batches already cover all pairs in {reg_path}; skipped")
            complete.add(judge)
        c = cal[judge]
        model = L.JUDGES[judge]["model"]
        # per request, so that long prompts get their own price tier; 15% margin on predicted input and mean output
        proj[judge] = float(sum(L.cost_usd(model, 1.15 * (c["a"] + c["b"] * x), 1.15 * c["tout"], batch=True) for x in chars))
        total += proj[judge]
    print(f"{len(pairs):,} pairs (mean query+passage {chars.mean():.0f} characters) x {len(args.judges)} judges")
    for judge in args.judges:
        c = cal[judge]
        print(f"  {judge}: projected ${proj[judge]:.2f} (tokens = {c['a']:.0f} + {c['b']:.3f} x chars, "
              f"output {c['tout']:.1f}; {c['n']} calls; +15% margin)")
    print(f"  total projected ${total:.2f}; limit --max-usd {args.max_usd}")
    if total > args.max_usd:
        sys.exit("projected cost exceeds --max-usd; nothing submitted")
    if args.dry_run:
        print("dry run: nothing submitted")
        return
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set in this shell.")
    client = anthropic.Anthropic(max_retries=0)  # creating a batch is not idempotent: never retry blindly
    for judge in [j for j in args.judges if j not in complete]:
        reqs = [Request(custom_id=L.custom_id(judge, r["qid"], r["docid"]),
                        params=MessageCreateParamsNonStreaming(**L.request_params(judge, r["query"], r["passage"])))
                for r in pairs]
        info = reg.get(judge) or dict(parts=[], n_requests=len(reqs), projected_usd=proj[judge],
                                      model=L.JUDGES[judge]["model"], prompt=L.JUDGES[judge]["prompt"],
                                      pairs=os.path.basename(args.pairs))
        done = sum(p["n_requests"] for p in info["parts"])  # resume after a failed part
        if done:
            print(f"  {judge}: resuming after {done:,} requests already submitted in {len(info['parts'])} parts")
        for start in range(done, len(reqs), args.part_size):
            part = reqs[start:start + args.part_size]
            try:
                b = client.messages.batches.create(requests=part)
            except Exception as e:
                recent = [(x.id, str(x.created_at), x.request_counts.processing) for x in client.messages.batches.list(limit=5)]
                sys.exit(f"creating part {start // args.part_size + 1} of {judge} failed ({type(e).__name__}: {e.__cause__!r}). The server may "
                         f"still have created it: check these recent batches before resubmitting: {recent}")
            info["parts"].append(dict(batch_id=b.id, start=start, n_requests=len(part)))
            reg[judge] = info
            with open(reg_path, "w") as fh:  # written after every part, so a crash cannot lose a batch id
                json.dump(reg, fh, indent=1)
            print(f"  submitted {judge} part {len(info['parts'])}: {b.id} ({len(part):,} requests)", flush=True)


def cmd_fetch(args):
    reg_path = os.path.join(args.dir, "batches.json")
    reg = json.load(open(reg_path))
    client = _client()
    total = 0.0
    for judge, info in reg.items():
        parts = info.get("parts") or [dict(batch_id=info["batch_id"])]  # old registries: one batch per judge
        bs = [client.messages.batches.retrieve(p["batch_id"]) for p in parts]
        rc = {k: sum(getattr(b.request_counts, k) for b in bs) for k in ("succeeded", "errored", "canceled", "expired", "processing")}
        status = "ended" if all(b.processing_status == "ended" for b in bs) else "in_progress"
        print(f"{judge}: {status} ({sum(b.processing_status == 'ended' for b in bs)}/{len(bs)} parts ended); "
              f"succeeded {rc['succeeded']}, errored {rc['errored']}, canceled {rc['canceled']}, expired {rc['expired']}, "
              f"processing {rc['processing']}")
        if status != "ended":
            continue
        model = info["model"]
        labels, failed, tin, tout, n, amb, c = defaultdict(dict), [], 0, 0, 0, 0, 0.0
        stops = Counter()
        with open(os.path.join(args.dir, f"{judge}_raw.jsonl"), "w") as fh:
            for res in (r for p in parts for r in client.messages.batches.results(p["batch_id"])):
                _, q, d = L.split_custom_id(res.custom_id)
                rec = dict(custom_id=res.custom_id, type=res.result.type)
                if res.result.type == "succeeded":
                    msg = res.result.message
                    text = L.text_of(msg.content)
                    lab = L.parse_label(text)
                    tin += msg.usage.input_tokens
                    tout += msg.usage.output_tokens
                    c += L.cost_usd(model, msg.usage.input_tokens, msg.usage.output_tokens, batch=True)
                    stops[msg.stop_reason] += 1
                    amb += L.ambiguous(text)
                    rec.update(text=text, label=lab, stop_reason=msg.stop_reason, ambiguous=L.ambiguous(text),
                               input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)
                    if lab is None:
                        failed.append(rec)
                    else:
                        labels[q][d] = lab
                else:
                    err = getattr(res.result, "error", None)
                    rec.update(error=json.dumps(_dump(err)) if err is not None else None)
                    failed.append(rec)
                n += 1
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        L.write_qrels(os.path.join(args.dir, f"{judge}.txt"), labels)
        L.write_jsonl(os.path.join(args.dir, f"{judge}_failed.jsonl"), failed)
        total += c
        print(f"  {n:,} results; labelled {sum(len(v) for v in labels.values()):,}; failed {len(failed)} "
              f"({len(failed) / max(n, 1):.2%}); two labels in one output {amb}; stop reasons {dict(stops)}; "
              f"tokens in {tin:,} out {tout:,}; "
              f"cost ${c:.2f} (projected ${info['projected_usd']:.2f})")
    print(f"actual cost of ended batches: ${total:.2f}")


# ---------------------------------------------------------------- Gate 2
def cmd_evaluate(args):
    data = D.load(args.data_dir)
    human = data["human"]
    pub = D.read_published(PUBLISHED)
    pub_tau = sorted(v["tau"] for v in pub.values())
    new = L.merge_label_dirs(args.dir, D.read_qrels)  # a later folder only fills missing labels (retries)
    n_h = sum(len(v) for v in human.values())
    print(f"published LLMJudge tau range {pub_tau[0]:.4f}-{pub_tau[-1]:.4f}; {n_h:,} human-judged pairs")
    tau = {}
    for j, q in new.items():
        cov = sum(d in q.get(t, {}) for t, docs in human.items() for d in docs)
        dist = Counter(v for docs in q.values() for v in docs.values())
        tau[j] = tau_on_test(data, q)
        print(f"  {j}: labels for {cov:,}/{n_h:,} human pairs; tau {tau[j]:.4f}; kappa {kappa(human, q):.4f}; "
              f"labels {dict(sorted(dist.items()))}")
    names = sorted(new)
    print("  pairwise: share of human-judged pairs with different labels; tau gap")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = sum(new[a].get(t, {}).get(d) != new[b].get(t, {}).get(d) for t, docs in human.items() for d in docs)
            print(f"    {a} vs {b}: {diff / n_h:.3f}; {abs(tau[a] - tau[b]):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selfcheck")
    p.add_argument("data_dir")
    p = sub.add_parser("pairs-pilot")
    p.add_argument("data_dir")
    p.add_argument("src_dir")
    p.add_argument("--out", required=True)
    p = sub.add_parser("ids")
    p.add_argument("data_dir")
    p.add_argument("--out", required=True)
    p = sub.add_parser("pairs-main")
    p.add_argument("data_dir")
    p.add_argument("--passages", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("sync")
    p.add_argument("pairs")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-usd", type=float, default=0.10)
    p.add_argument("--judges", nargs="+", default=[j for j in L.JUDGES if L.JUDGES[j].get("api", "anthropic") == "anthropic"])
    p = sub.add_parser("submit")
    p.add_argument("pairs")
    p.add_argument("--calib", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-usd", type=float, required=True)
    p.add_argument("--judges", nargs="+", default=[j for j in L.JUDGES if L.JUDGES[j].get("api", "anthropic") == "anthropic"])
    p.add_argument("--calib-pairs", nargs="+", help="pair files covering the calibration calls (default: PAIRS)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--part-size", type=int, default=2000, help="requests per batch (a 58.8 MB upload failed)")
    p = sub.add_parser("fetch")
    p.add_argument("dir")
    p = sub.add_parser("run")
    p.add_argument("pairs")
    p.add_argument("--out", required=True)
    p.add_argument("--max-usd", type=float, required=True)
    p.add_argument("--judges", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retry-failed", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("data_dir")
    p.add_argument("dir", nargs="+")
    args = ap.parse_args()
    for j in getattr(args, "judges", []) or []:
        if j not in L.JUDGES:
            sys.exit(f"unknown judge {j}; known: {', '.join(L.JUDGES)}")
        try:  # before any call: the LLMJudge sample prompt is read from a file
            L.template(L.JUDGES[j]["prompt"])
        except L.PromptFileError as e:
            sys.exit(f"{j}: {e}")
    dict(selfcheck=cmd_selfcheck, **{"pairs-pilot": cmd_pairs_pilot, "pairs-main": cmd_pairs_main}, ids=cmd_ids,
         sync=cmd_sync, run=cmd_run, submit=cmd_submit, fetch=cmd_fetch, evaluate=cmd_evaluate)[args.cmd](args)


if __name__ == "__main__":
    main()
