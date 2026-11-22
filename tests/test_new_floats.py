"""Steps 35 and 36: the resolution table and the held-out figure agree with the released result files."""
import importlib.util
import itertools
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _summary():
    path = os.path.join(ROOT, "results", "llm-judges", "summary.tsv")
    with open(path) as f:
        next(f)
        return dict(line.rstrip("\n").split("\t")[:2] for line in f if line.strip())


def test_tau_by_pool_matches_summary():
    pytest.importorskip("matplotlib")
    m = _load("36_heldout_figure")
    T = m.tau_by_pool(os.path.join(ROOT, "results", "llm-judges", "29_power_bootstrap.tsv"))
    S = _summary()
    for pool in ("test", "heldout"):
        assert round(min(T[pool].values()), 4) == float(S[f"tau_min_{pool}"])
        assert round(max(T[pool].values()), 4) == float(S[f"tau_max_{pool}"])
    judges = sorted(T["test"])
    flips = sum((T["test"][a] - T["test"][b]) * (T["heldout"][a] - T["heldout"][b]) < 0
                for a, b in itertools.combinations(judges, 2))
    assert flips == int(S["order_flips_test_heldout"])
    assert min(judges, key=lambda j: T["test"][j]) == S["bottom_test"] == S["top_heldout"]


def test_resolution_table_cells(tmp_path):
    m = _load("35_make_resolution_table")
    assert m.cell("none", "0") == "none"
    assert m.cell("0.116254", "11") == "0.116 (11)"
    assert m.pct("1.000000") == "100" and m.pct("0.932000") == "93.2"
    out = tmp_path / "resolution.tex"
    import sys
    argv = sys.argv
    sys.argv = ["35", "--out", str(out)]
    try:
        m.main()
    finally:
        sys.argv = argv
    tex = out.read_text()
    assert "200 & 0.116 (11) & none & none & 93.2" in tex
    assert "400 & 0.087 (31) & 0.116 (11) & none" in tex
