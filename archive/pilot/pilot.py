"""Smoke-test pilot: LLMJudge (TREC DL 2023, 25 test topics) judges vs human qrels.
Per-topic nDCG@10 for 35 DL-2023 passage runs under human qrels and each of 33 public
LLMJudge label sets; Kendall tau-b of system rankings; paired topic bootstrap CIs; and a
paired topic-level permutation test for adjacent judges."""
import sys, os, math, random
from collections import defaultdict

D = sys.argv[1]
random.seed(0)

def read_map(p):
    m = {}
    for line in open(p):
        a, b = line.split()
        m[b] = a
    return m

qmap = read_map(os.path.join(D, "qid_map.txt"))
dmap = read_map(os.path.join(D, "docid_map.txt"))

def read_qrels(p, mapped):
    q = defaultdict(dict)
    for line in open(p):
        f = line.split()
        if len(f) < 4:
            continue
        qid, did, rel = f[0], f[2], f[3]
        if mapped:
            qid, did = qmap[qid], dmap[did]
        q[qid][did] = max(0, int(float(rel)))
    return q

human = read_qrels(os.path.join(D, "human_ids.txt"), False)
topics = sorted(human)
judges = {}
for fn in sorted(os.listdir(os.path.join(D, "judges"))):
    judges[fn.replace(".txt", "")] = read_qrels(os.path.join(D, "judges", fn), True)

runs = {}
for fn in sorted(os.listdir(os.path.join(D, "runs"))):
    r = defaultdict(list)
    for line in open(os.path.join(D, "runs", fn)):
        f = line.split()
        if f[0] in human:
            r[f[0]].append((float(f[4]), f[2]))
    top = {}
    for q, lst in r.items():
        lst.sort(key=lambda x: (-x[0], x[1]), reverse=False)
        # trec_eval: score desc, docno desc on ties
        lst.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top[q] = [d for _, d in lst[:10]]
    runs[fn] = top
systems = sorted(runs)

def ndcg10(ranked, qrel):
    dcg = sum((2 ** qrel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked))
    ideal = sorted(qrel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def per_topic(qrels):
    return {s: [ndcg10(runs[s].get(q, []), qrels.get(q, {})) for q in topics] for s in systems}

H = per_topic(human)
J = {j: per_topic(qr) for j, qr in judges.items()}

def taub(x, y):
    nc = nd = tx = ty = 0
    n = len(x)
    for i in range(n):
        for k in range(i + 1, n):
            a = x[i] - x[k]; b = y[i] - y[k]
            if a == 0 and b == 0:
                continue
            if a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif (a > 0) == (b > 0):
                nc += 1
            else:
                nd += 1
    den = math.sqrt((nc + nd + tx) * (nc + nd + ty))
    return (nc - nd) / den if den else 0.0

def means(M, idx):
    return [sum(M[s][i] for i in idx) / len(idx) for s in systems]

allidx = list(range(len(topics)))
hm = means(H, allidx)
tau = {j: taub(hm, means(J[j], allidx)) for j in J}
order = sorted(tau, key=lambda j: -tau[j])

B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
boots = {j: [] for j in J}
for _ in range(B):
    idx = [random.randrange(len(topics)) for _ in topics]
    h = means(H, idx)
    for j in J:
        boots[j].append(taub(h, means(J[j], idx)))

def ci(v, a=0.05):
    v = sorted(v); n = len(v)
    return v[int(a / 2 * n)], v[int((1 - a / 2) * n) - 1]

print(f"topics={len(topics)} systems={len(systems)} judges={len(J)} B={B}")
print("rank judge tau 95%CI")
for r, j in enumerate(order, 1):
    lo, hi = ci(boots[j])
    print(f"{r:2d} {j:28s} {tau[j]:.4f} [{lo:.3f}, {hi:.3f}]")

# adjacent pairs: paired bootstrap CI on delta, and permutation test (swap judges per topic)
P = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
print("\nadjacent pairs: delta, paired-bootstrap 95% CI of delta, permutation p (two-sided)")
sig = 0
for a, b in zip(order, order[1:]):
    d = tau[a] - tau[b]
    dl = [x - y for x, y in zip(boots[a], boots[b])]
    lo, hi = ci(dl)
    cnt = 0
    for _ in range(P):
        A = {s: [] for s in systems}; Bm = {s: [] for s in systems}
        swap = [random.random() < 0.5 for _ in topics]
        for s in systems:
            A[s] = [J[b][s][i] if swap[i] else J[a][s][i] for i in allidx]
            Bm[s] = [J[a][s][i] if swap[i] else J[b][s][i] for i in allidx]
        dd = taub(hm, means(A, allidx)) - taub(hm, means(Bm, allidx))
        if abs(dd) >= abs(d) - 1e-12:
            cnt += 1
    p = (cnt + 1) / (P + 1)
    sig += p < 0.05
    print(f"{a:28s} vs {b:28s} d={d:.4f} CI=[{lo:.3f},{hi:.3f}] p={p:.3f}")
print(f"\nadjacent pairs with p<0.05: {sig}/{len(order)-1}")
top, last = order[0], order[-1]
print("best vs worst delta", round(tau[top] - tau[last], 4))
