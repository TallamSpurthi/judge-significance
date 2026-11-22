"""Tests for LLM labelling (no network): prompts, parser, request bodies, ids, cost, qrels I/O, corpus extraction."""
import gzip
import hashlib
import importlib.util
import io
import json
import os
import random
import tarfile

import numpy as np
import pytest

from judge_significance import data as D
from judge_significance import llm_label as L

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("x27", os.path.join(HERE, "..", "scripts", "27_extract_passages.py"))
X27 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X27)

# sha256 of the prompt templates, checked byte for byte against their sources on Nov 9, 2026
PROMPT_SHA = {"umbrela": "c92537a4f97293df", "basic": "20e78e821fb9e57a"}  # sources
# the LLMJudge sample prompt is not stored in this repository; its template is tested when LLMJUDGE_PROMPT is set
needs_prompt = pytest.mark.skipif(not os.environ.get("LLMJUDGE_PROMPT"), reason="LLMJUDGE_PROMPT is not set")
TEMPLATES = ["umbrela", "basic", pytest.param("llmjudge", marks=needs_prompt)]


def test_prompt_templates_unchanged():
    sources = {"umbrela": L.UMBRELA_BING, "basic": L.UMBRELA_BASIC}
    for k, v in sources.items():  # the verbatim source texts
        assert hashlib.sha256(v.encode()).hexdigest()[:16] == PROMPT_SHA[k], k
    assert L.PROMPTS == {"umbrela": L.UMBRELA_BING + L.UMBRELA_ADDED, "basic": L.UMBRELA_BASIC}  # only the marked addition
    assert L.UMBRELA_BING.splitlines()[1].endswith("query, ")  # trailing spaces of the source kept


@needs_prompt
def test_llmjudge_prompt_is_the_file_plus_one_line():
    sample = L.llmjudge_sample(os.environ["LLMJUDGE_PROMPT"])
    assert hashlib.sha256(sample.encode()).hexdigest() == L.LLMJUDGE_PROMPT_SHA256 and len(sample) == 1244
    tpl = L.template("llmjudge")
    assert tpl == sample + L.LLMJUDGE_ADDED
    # the full template used for the released labels, added line included
    assert hashlib.sha256(tpl.encode()).hexdigest()[:16] == "ccaa400376215bac"


def test_llmjudge_prompt_missing_or_changed(tmp_path, monkeypatch):
    other = tmp_path / "other.txt"
    other.write_text("a different prompt {query} {passage}\n")
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe")
    for path, message in (("", "set LLMJUDGE_PROMPT"), (str(tmp_path / "none.txt"), "cannot read"),
                          (str(binary), "cannot read"), (str(other), "SHA-256 differs")):
        monkeypatch.setenv("LLMJUDGE_PROMPT", path)
        with pytest.raises(L.PromptFileError, match=message):
            L.render("llmjudge", "q", "p")
        with pytest.raises(L.PromptFileError):
            L.request_params("haiku-llmjudge", "q", "p")
        assert L.render("umbrela", "q", "p") and L.render("basic", "q", "p")  # the other prompts need no file
    monkeypatch.delenv("LLMJUDGE_PROMPT")
    with pytest.raises(L.PromptFileError, match="set LLMJUDGE_PROMPT"):
        L.template("llmjudge")


def test_labelling_stops_before_any_call_without_the_prompt(monkeypatch):
    monkeypatch.delenv("LLMJUDGE_PROMPT", raising=False)
    argv = ["28_llm_label.py", "run", "missing.jsonl", "--out", "missing", "--max-usd", "0", "--judges", "haiku-llmjudge"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="haiku-llmjudge: set LLMJUDGE_PROMPT"):
        _x28().main()


@pytest.mark.parametrize("p", TEMPLATES)
def test_render_fills_placeholders_and_keeps_braces(p):
    s = L.render(p, " what is {x}? ", "a {passage} with braces {query}")
    assert "what is {x}?" in s and "a {passage} with braces {query}" in s
    assert s.count("{query}") == 1 and s.count("{passage}") == 1  # only the ones inside the passage


@pytest.mark.parametrize("text,label", [
    ("##final score: 2", 2), ("## final score: 3", 3), ("##Final Score: 0\n", 0), ("M: 2\nT: 1\n##final score: 1", 1),
    ("Final score (O): 2", 2), ("##final score: **3**", 3), ("2", 2), ("  3  ", 3), ("Score: 1", 1), ("**2**", 2),
    ("The passage explains the cause.\n\n2", 2), ("Explanation: related only.\nCategory: 1", 1), ("Relevance category: 0", 0),
    ("final score: 2\nfinal score: 3", 3), ("2.", 2),
])
def test_parse_label(text, label):
    assert L.parse_label(text) == label


@pytest.mark.parametrize("text", ["", "   ", "4", "Score: 5", "2 or 3", "The answer is 2 because 3 facts", "1.5",
                                  "final score: 2.5", "I cannot judge this.", "score: 23"])
def test_parse_label_rejects(text):
    assert L.parse_label(text) is None


def test_request_params_per_model():
    p = L.request_params("haiku-umbrela", "q", "p")
    assert p["model"] == "claude-haiku-5-5" and p["thinking"] == {"type": "disabled"} and p["output_config"] == {"effort": "low"}
    s = L.request_params("sonnet-umbrela", "q", "p")
    assert s["model"] == "claude-sonnet-5-5" and s["thinking"] == {"type": "between_tools"}
    assert "temperature" not in p and "temperature" not in s  # rejected by both models
    p["thinking"]["type"] = "changed"
    assert L.JUDGES["haiku-umbrela"]["thinking"] == {"type": "disabled"}  # params are copies


def test_custom_id_roundtrip():
    cid = L.custom_id("haiku-llmjudge", "2001010", "msmarco_passage_00_257661787")
    assert len(cid) <= 64 and L.split_custom_id(cid) == ("haiku-llmjudge", "2001010", "msmarco_passage_00_257661787")
    with pytest.raises(ValueError):
        L.custom_id("bad judge", "1", "d")


def test_text_of_skips_thinking():
    content = [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "##final score: "}, {"type": "text", "text": "2"}]
    assert L.text_of(content) == "##final score: 2"


def test_cost():
    assert L.cost_usd("claude-haiku-5-5", 1_000, 1_000, batch=False) == pytest.approx(0.0006)
    assert L.cost_usd("claude-sonnet-5-5", 1_000_000, 1_000_000, batch=True) == pytest.approx(6.0)
    # Haiku prompts above 100K tokens use the higher tier, per request
    assert L.cost_usd("claude-haiku-5-5", 100_000, 0, batch=False) == pytest.approx(0.01)
    assert L.cost_usd("claude-haiku-5-5", 200_000, 1_000, batch=False) == pytest.approx(0.1 + 0.0025)
    assert L.cost_usd("claude-sonnet-5-5", 200_000, 0, batch=False) == pytest.approx(0.4)


def test_qrels_roundtrip(tmp_path):
    lab = {"2001010": {"msmarco_passage_00_1": 2, "msmarco_passage_00_0": 0}, "300": {"d": 3}}
    L.write_qrels(tmp_path / "q.txt", lab)
    assert D.read_qrels(tmp_path / "q.txt") == lab


# ---------------------------------------------------------------- corpus extraction
def _bundle(bb, n, noise=0):
    """Uncompressed bundle lines whose pids encode their byte offsets, as in MS MARCO v2. noise adds that many
    random hex characters per line, so the bundle does not compress (gzip reads its input in large chunks)."""
    rnd = random.Random(0)
    lines, pos, pids = [], 0, []
    for i in range(n):
        pid = f"msmarco_passage_{bb}_{pos}"
        pad = "%0*x" % (noise, rnd.getrandbits(4 * noise)) if noise else ""
        line = (json.dumps({"pid": pid, "passage": f"text {bb} {i} " + "x" * (i % 7) + pad, "spans": "", "docid": "d"}) + "\n").encode()
        lines.append(line)
        pids.append(pid)
        pos += len(line)
    return b"".join(lines), pids


def _tar(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, payload in members:
            ti = tarfile.TarInfo(name)
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
    return buf.getvalue()


def test_extract_selects_by_offset_and_checks_pid():
    b0, p0 = _bundle("00", 50)
    b1, p1 = _bundle("01", 30)
    tar = _tar([("msmarco_v2_passage/msmarco_passage_00.gz", gzip.compress(b0)),
                ("msmarco_v2_passage/README.txt", b"readme"),
                ("msmarco_v2_passage/msmarco_passage_01.gz", gzip.compress(b1))])
    wanted = [p0[0], p0[17], p0[49], p1[5]]
    want, bad = X27.needed_offsets(wanted + ["not_a_pid"])
    assert bad == ["not_a_pid"]
    reader = X27.HashingReader(io.BytesIO(tar))
    got = dict(X27.extract(reader, want))
    while reader.read(1 << 16):
        pass
    assert sorted(got) == sorted(wanted)
    assert got[p0[17]]["passage"].startswith("text 00 17")
    assert reader.md5.hexdigest() == hashlib.md5(tar).hexdigest() and reader.n == len(tar)


def test_extract_rejects_wrong_offset():
    b0, p0 = _bundle("00", 5)
    tar = _tar([("x/msmarco_passage_00.gz", gzip.compress(b0))])
    shifted = int(p0[2].rsplit("_", 1)[1])
    want = {"00": {shifted: "msmarco_passage_00_999"}}  # an offset that holds a different pid
    with pytest.raises(ValueError):
        list(X27.extract(X27.HashingReader(io.BytesIO(tar)), want))


def test_extract_truncated_stream_stops_cleanly():
    b0, p0 = _bundle("00", 20000, noise=100)  # about 2 MB, so half the stream spans many gzip reads
    tar = _tar([("x/msmarco_passage_00.gz", gzip.compress(b0, compresslevel=1))])
    want, _ = X27.needed_offsets([p0[0], p0[-1]])
    got = dict(X27.extract(X27.HashingReader(io.BytesIO(tar), max_bytes=len(tar) // 2), want))
    assert p0[0] in got and p0[-1] not in got


def test_ambiguous():
    assert L.ambiguous("Explanation...\n\n3\nAnswer: 1") and L.parse_label("Explanation...\n\n3\nAnswer: 1") == 1
    assert not L.ambiguous("##final score: 2") and not L.ambiguous("It mentions 3 facts.\n2")
    assert L.ambiguous("##final score: 1\n##final score: 2")


def _x28():
    spec = importlib.util.spec_from_file_location("x28", os.path.join(HERE, "..", "scripts", "28_llm_label.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_resumes_and_retries_failed_once(tmp_path, monkeypatch):
    """`run` skips answered pairs, re-asks unparsed ones once with --retry-failed, and writes fetch-style outputs."""
    from types import SimpleNamespace as NS
    x28 = _x28()
    replies = {"d1": ["##final score: 2"], "d2": ["I think", "still thinking", "##final score: 3"]}
    calls = []

    def create(**params):
        doc = params["messages"][0]["content"].split("PASSAGE-")[1].split()[0]
        calls.append(doc)
        text = replies[doc].pop(0)
        return NS(content=[NS(type="text", text=text)], usage=NS(input_tokens=500, output_tokens=8),
                  stop_reason="end_turn", id="msg_x", model=L.HAIKU)

    monkeypatch.setattr(x28, "_client", lambda: NS(messages=NS(create=create)))
    pairs = tmp_path / "pairs.jsonl"
    L.write_jsonl(pairs, [dict(qid="q1", docid=d, query="what", passage=f"PASSAGE-{d} text") for d in ("d1", "d2")])
    out = tmp_path / "out"
    args = NS(pairs=str(pairs), out=str(out), judges=["haiku-umbrela"], max_usd=1.0, workers=2, retry_failed=False)
    x28.cmd_run(args)
    assert sorted(calls) == ["d1", "d2"]
    x28.cmd_run(args)  # resume: nothing re-asked
    assert len(calls) == 2
    args.retry_failed = True
    x28.cmd_run(args)  # d2 re-asked once (still unparsed)
    x28.cmd_run(args)  # attempt limit reached: not re-asked again
    assert sorted(calls) == ["d1", "d2", "d2"]
    assert D.read_qrels(str(out / "haiku-umbrela.txt")) == {"q1": {"d1": 2}}
    failed = L.read_jsonl(out / "haiku-umbrela_failed.jsonl")
    assert [(r["custom_id"], r["attempt"]) for r in failed] == [(L.custom_id("haiku-umbrela", "q1", "d2"), 2)]


def test_run_stops_at_max_usd(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    x28 = _x28()
    monkeypatch.setattr(x28, "_client", lambda: NS(messages=NS(create=lambda **p: NS(
        content=[NS(type="text", text="##final score: 1")], usage=NS(input_tokens=1_000_000, output_tokens=0),
        stop_reason="end_turn", id="m", model=L.HAIKU))))
    pairs = tmp_path / "pairs.jsonl"
    L.write_jsonl(pairs, [dict(qid="q", docid=f"d{i}", query="a", passage="b") for i in range(5)])
    args = NS(pairs=str(pairs), out=str(tmp_path / "o"), judges=["haiku-umbrela"], max_usd=0.25, workers=1,
              retry_failed=False)
    x28.cmd_run(args)  # the cap is checked before each call, so it stops after the first call that reaches it
    n = len(L.read_jsonl(tmp_path / "o" / "haiku-umbrela_raw.jsonl"))
    per = L.cost_usd(L.HAIKU, 1_000_000, 0, batch=False)
    assert n == int(np.ceil(0.25 / per)) and n < 5


def test_merge_label_dirs_later_folder_only_fills(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    L.write_qrels(str(a / "haiku-umbrela.txt"), {"q": {"d1": 2}})
    L.write_qrels(str(b / "haiku-umbrela.txt"), {"q": {"d1": 0, "d2": 3}})
    L.write_qrels(str(b / "notajudge.txt"), {"q": {"d1": 1}})
    assert L.merge_label_dirs([str(a), str(b)], D.read_qrels) == {"haiku-umbrela": {"q": {"d1": 2, "d2": 3}}}


def test_openrouter_body_pins_endpoint_and_disables_compression():
    for judge, j in L.OPEN_JUDGES.items():
        b = L.openrouter_body(judge, "q?", "a passage")
        assert b["model"] == j["model"] and b["temperature"] == 0.0 and b["seed"] == 0 and b["max_tokens"] == 32
        assert b["provider"] == dict(order=[j["provider"]], allow_fallbacks=False, require_parameters=True,
                                     quantizations=[j["quantization"]], data_collection="deny")
        assert b["plugins"] == [{"id": "context-compression", "enabled": False}]
        assert b["messages"][0]["content"] == L.render("umbrela", "q?", "a passage")
        assert L.custom_id(judge, "2001", "msmarco_passage_00_1")
    with pytest.raises(ValueError):
        L.request_params("llama8b-umbrela", "q", "p")


def test_openrouter_call_success_retry_and_error(monkeypatch):
    import urllib.error
    x28 = _x28()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-key")
    monkeypatch.setattr("time.sleep", lambda s: None)
    ok = {"id": "gen-1", "model": "meta-llama/llama-3.1-8b-instruct", "provider": "DeepInfra",
          "choices": [{"message": {"content": "##final score: 2"}, "finish_reason": "stop"}],
          "usage": {"prompt_tokens": 400, "completion_tokens": 7, "cost": 0.00001}}
    script = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout):
        assert req.get_header("Authorization") == "Bearer test-not-a-key"
        body = json.loads(req.data)
        assert body["provider"]["allow_fallbacks"] is False
        kind = script.pop(0)
        if isinstance(kind, int):
            raise urllib.error.HTTPError(req.full_url, kind, "x", {}, io.BytesIO(b'{"error":{"code":%d}}' % kind))
        return Resp(json.dumps(kind).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    script[:] = [429, 503, ok]
    r = x28._openrouter_call("llama8b-umbrela", "q", "p")
    assert r["type"] == "succeeded" and r["label"] == 2 and r["cost_usd"] == 0.00001 and r["provider"] == "DeepInfra"
    assert script == []
    script[:] = [400, ok]  # context overflow: no retry
    r = x28._openrouter_call("llama8b-umbrela", "q", "p")
    assert r["type"] == "errored" and r["error"].startswith("HTTP 400") and script == [ok]
    script[:] = [{"error": {"code": 502, "message": "upstream"}}]  # error inside a 200 body
    r = x28._openrouter_call("llama8b-umbrela", "q", "p")
    assert r["type"] == "errored" and r["label"] is None


def test_passage_cut_only_for_short_context_judges():
    long = "x" * 300_000
    assert L.passage_for("llama8b-umbrela", long) == ("x" * 280_000, True)
    assert L.passage_for("llama8b-umbrela", "short ") == ("short", False)
    assert L.passage_for("qwen30b-umbrela", long) == (long, False)
    assert L.passage_for("sonnet-umbrela", long) == (long, False)
    prompt_chars = len(L.render("umbrela", "q", ""))
    assert len(L.openrouter_body("gemma4b-umbrela", "q", long)["messages"][0]["content"]) == prompt_chars + 280_000


def test_submit_in_parts_resumes_and_fetch_merges(tmp_path, monkeypatch):
    """submit splits a judge into parts, never retries a failed create, resumes after it, and fetch merges parts."""
    from types import SimpleNamespace as NS
    x28 = _x28()
    pairs = tmp_path / "pairs.jsonl"
    L.write_jsonl(pairs, [dict(qid="q", docid=f"d{i}", query="a", passage="b" * (10 + 50 * i)) for i in range(5)])
    calib = tmp_path / "calib"
    calib.mkdir()
    L.write_jsonl(calib / "haiku-umbrela_raw.jsonl", [dict(custom_id=L.custom_id("haiku-umbrela", "q", f"d{i}"),
                  type="succeeded", input_tokens=400 + 30 * i, output_tokens=8) for i in range(5)] * 3)
    created, fail_on = {}, {2}
    counts = NS(succeeded=0, errored=0, canceled=0, expired=0, processing=0)

    def create(requests):
        if len(created) + 1 in fail_on:
            fail_on.clear()
            raise ConnectionError("upload broke")
        bid = f"b{len(created) + 1}"
        created[bid] = list(requests)
        return NS(id=bid)

    def results(bid):
        for r in created[bid]:
            msg = NS(content=[NS(type="text", text="##final score: 1")], usage=NS(input_tokens=400, output_tokens=8),
                     stop_reason="end_turn")
            yield NS(custom_id=r["custom_id"], result=NS(type="succeeded", message=msg))

    batches = NS(create=create, list=lambda limit: [], results=results,
                 retrieve=lambda bid: NS(processing_status="ended", request_counts=counts))
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: NS(messages=NS(batches=batches)))
    monkeypatch.setattr(x28, "_client", lambda: NS(messages=NS(batches=batches)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    out = tmp_path / "out"
    args = NS(pairs=str(pairs), calib=str(calib), calib_pairs=None, out=str(out), max_usd=1.0, judges=["haiku-umbrela"],
              dry_run=False, part_size=2)
    with pytest.raises(SystemExit):
        x28.cmd_submit(args)  # part 2 fails: stop, keep part 1 registered
    reg = json.load(open(out / "batches.json"))
    assert [p["batch_id"] for p in reg["haiku-umbrela"]["parts"]] == ["b1"]
    x28.cmd_submit(args)  # resumes with the remaining 3 requests
    reg = json.load(open(out / "batches.json"))
    assert [(p["start"], p["n_requests"]) for p in reg["haiku-umbrela"]["parts"]] == [(0, 2), (2, 2), (4, 1)]
    n_created = len(created)
    x28.cmd_submit(args)  # complete: skipped, nothing created
    assert len(created) == n_created
    x28.cmd_fetch(NS(dir=str(out)))
    assert D.read_qrels(str(out / "haiku-umbrela.txt")) == {"q": {f"d{i}": 1 for i in range(5)}}
