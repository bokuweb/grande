"""Rank omg JevBench runs among the published JevBench v1.2 systems on the 231 public items,
and estimate a JevBench Score with the official formulas (composite_v12)."""
import json, sys, math, glob, os
sys.path.insert(0, sys.argv[1])  # usage: jevbench_compare.py <jevbench checkout> <runs dir with <model>/<tier>/results.jsonl>
from jevbench import composite_v12 as c
J = sys.argv[1]; RUNS = sys.argv[2]
d = json.load(open(f'{J}/results/v1.2/jevbench-v1.2-per-task.json'))
tasks = {t['id']: t for t in d['tasks']}
pub = {}
for f in ('easy', 'original', 'hard'):
    for l in open(f'{J}/datasets/public/{f}.jsonl'):
        t = json.loads(l); pub[t['id']] = t
tiers = {k: [i for i, t in tasks.items() if t['tier'] == k] for k in ('easy', 'standard', 'hard')}
def acc(pt, ids): return sum(pt[i][0] == 'c' for i in ids if i in pt) / len(ids)
def Ipub(e, s, h): return 100 * (0.14 * e + 0.28 * s + 0.30 * h) / 0.72
def ece(rows, bins=10):
    b = [[] for _ in range(bins)]
    for conf, ok in rows: b[min(bins - 1, int(conf * bins))].append((conf, ok))
    n = len(rows); return sum(len(x) / n * abs(sum(o for _, o in x) / len(x) - sum(cf for cf, _ in x) / len(x)) for x in b if x)
rows = []
for name, s in d['systems'].items():
    pt = s['public_tasks']
    rows.append((s['display'][:46] + (' (partial)' if s['partial'] else ''), acc(pt, tiers['easy']), acc(pt, tiers['standard']), acc(pt, tiers['hard']), s['by_tier'].get('intelligence')))
mine = {}
for run in sorted(glob.glob(f'{RUNS}/*/hard/results.jsonl')):
    M = run.split('/')[-3]; pt = {}; lat = {'standard': [], 'hard': []}; hard_rows = []; tvds = []
    for f, tier in (('easy', 'easy'), ('original', 'standard'), ('hard', 'hard')):
        p = f'{RUNS}/{M}/{f}/results.jsonl'
        if not os.path.exists(p): continue
        for l in open(p):
            r = json.loads(l); pt[r['task_id']] = ('c' if r['correct'] else 'w', 0)
            if tier in lat: lat[tier].append(r['latency_s'] if 'latency_s' in r else r.get('latency'))
            if tier == 'hard':
                probs = r['probs']; top = max(probs, key=probs.get); hard_rows.append((probs[top], 1.0 if r['correct'] else 0.0))
                gp = pub[r['task_id']]['provenance'].get('gold_probs')
                if gp: tvds.append(c.tvd(probs, gp, pub[r['task_id']]['labels']))
    if len(pt) < 231: continue
    e, s_, h = acc(pt, tiers['easy']), acc(pt, tiers['standard']), acc(pt, tiers['hard'])
    rows.append((f'>> omg {M} zero-shot (M4 Metal)', e, s_, h, None))
    # the official p50/p95 come from the standard+judge run; judge is not public, so standard only
    l = sorted(x for x in lat['standard'] if x is not None)
    p50, p95 = l[len(l) // 2], l[int(len(l) * 0.95)]
    mine[M] = dict(I=c.intelligence({'easy': e, 'standard': s_, 'judge': s_ - 0.03, 'hard': h}), ece=ece(hard_rows), tvd=sum(tvds) / len(tvds) if tvds else None, p50=p50, p95=p95)
rows.sort(key=lambda r: -Ipub(r[1], r[2], r[3]))
print(f"{'#':>2} {'system (same 231 public items)':58} {'easy':>5} {'std':>5} {'hard':>5} {'I(pub)':>6} {'I(official)':>11}")
for n, r in enumerate(rows, 1):
    print(f"{n:2} {r[0]:58} {r[1]*100:5.1f} {r[2]*100:5.1f} {r[3]*100:5.1f} {Ipub(r[1],r[2],r[3]):6.1f} {'' if r[4] is None else f'{r[4]:11.1f}'}")
print("\nEstimated JevBench Score for omg (official formulas). Judge tier (not public) proxied as standard - 3 pts; speed from the\n"
      "standard tier as our own server (x2 + 0.15 s); cost = hosted-provider estimate for the size class; calibration raw (T = 1).")
for M, m in mine.items():
    for kind, usd in (('cpu', 0.016 if 'E2B' in M else 0.023),):
        I = m['I']; C = c.calibration(m['ece'], m['tvd']); S = c.speed(m['p50'], m['p95'], kind); K = c.cost(usd)
        score = c.jevbench_score(dict(intelligence=I, calibration=C, speed=S, cost=K))
        print(f"  {M:28} I {I:5.1f} (judge proxied)  C {C:5.1f} (ECE {m['ece']:.3f}, TVD {m['tvd']:.3f})  S {S:5.1f} (p50 {m['p50']:.2f}s p95 {m['p95']:.2f}s raw)  K {K:5.1f} (~${usd}/1k)  => score {score:5.1f}")
