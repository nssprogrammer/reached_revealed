# ---- Paper figures + headline numbers (CPU, seconds). Inputs: swe_out.tgz (SWE-rebench v3), gemma_out.tgz (E1), gemma_eval.tar.gz (E2),
# extracted anywhere under DATA_ROOT. Output: FIG_DIR/fig1..fig4.png and numbers.json.
import os, glob, json, tarfile, random, zlib
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DATA_ROOT = Path(os.environ.get('DATA_ROOT', '/kaggle/input')); FIG = Path(os.environ.get('FIG_DIR', '/kaggle/working/figures')); FIG.mkdir(parents=True, exist_ok=True)
for tgz in glob.glob(str(DATA_ROOT / '**/*.t*gz'), recursive=True):              # extract any attached archive next to the figures
    dst = FIG.parent / ('x_' + Path(tgz).name.split('.')[0])
    if not dst.exists():
        with tarfile.open(tgz) as tf: tf.extractall(dst, filter='data')
ROOTS = [DATA_ROOT, FIG.parent]
def g(pat): return sorted({p for r in ROOTS for p in glob.glob(str(r / '**' / pat), recursive=True)})

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE, AQUA, YELLOW, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#b9b8b2'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'savefig.facecolor': SURF, 'font.family': 'DejaVu Sans'})
rng = np.random.default_rng(0); NUM = {}
PAPER = os.environ.get('PAPER') == '1'                      # column-width figures for the IEEE paper: no titles (captions carry them)
W = 3.45 if PAPER else 6.4; FS = 6.5 if PAPER else 10; FS2 = 6 if PAPER else 9
if PAPER: plt.rcParams.update({'font.size': 7, 'legend.fontsize': 6.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5})
_sup = plt.Figure.suptitle
def _suptitle(self, *a, **k):
    return None if PAPER else _sup(self, *a, **k)
plt.Figure.suptitle = _suptitle
def boot(df, fn, by, B=4000):
    grp = {k: q for k, q in df.groupby(by)}; ks = list(grp); v = []
    for _ in range(B): v.append(fn(pd.concat([grp[k] for k in rng.choice(ks, len(ks))])))
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]

# ------------------------------------------------------------ SWE-rebench v3
RES = Path(os.environ.get('RESULTS_DIR', Path(__file__).resolve().parents[2] / 'results' if '__file__' in globals() else 'results'))
_sum = g('swe/out/*/summary.json')                       # raw run outputs if attached, else the tables shipped in results/
S = pd.DataFrame([json.load(open(f)) for f in _sum]) if _sum else pd.read_csv(RES / 'swe_rebench_tasks.csv')
S = S[S.get('v', 0) == 3]; R = S[S.status == 'reproduced'].copy()
cov = R[R.old_tests_hit_F > 0]
NUM['swe'] = dict(candidates=len(S), reproduced=len(R), repos=int(R.repo.nunique()), executed=len(cov), silent=int((cov.d_old == 0).sum()),
                  silent_ci=boot(cov, lambda q: (q.d_old == 0).mean(), 'repo'), median_old_tests=float(cov.old_tests_hit_F.median()),
                  funnel=S.status.value_counts().to_dict())
fig, ax = plt.subplots(figsize=(W, 1.75 if PAPER else 2.6))
x = cov.old_tests_hit_F.values.astype(float); det = cov.d_old.values > 0; y = rng.uniform(-0.3, 0.3, len(x))
ax.scatter(x[~det], y[~det], s=22, color=BLUE, alpha=0.75, edgecolor=SURF, linewidth=0.6, label=f'no old test fails ({(~det).sum()})')
ax.scatter(x[det], y[det], s=60, color=ORANGE, marker='D', edgecolor=SURF, linewidth=0.8, label=f'an old test fails ({det.sum()})')
ax.set_xscale('log'); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_ylim(-0.6, 0.9)
ax.set_xlabel('pre-existing tests that execute the buggy function (log scale)'); ax.grid(axis='x', color=GRID, lw=0.6)
ax.axvline(np.median(x), color=INK2, lw=1, ls=':'); ax.text(np.median(x) * 1.08, 0.72, f'median {np.median(x):.0f}', color=INK2, fontsize=FS2)
ax.legend(loc='upper center', frameon=False, fontsize=6.5 if PAPER else 9, ncol=2, bbox_to_anchor=(0.5, -0.42))
fig.suptitle(f'Real bugs are reached but not revealed ({len(cov)} SWE-rebench tasks, {cov.repo.nunique()} repositories)', x=0.01, ha='left', fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(FIG / 'fig1_reach_not_reveal.png', dpi=300, bbox_inches='tight', pad_inches=0.03); plt.close(fig)

M = R.dropna(subset=['caught_any']).copy()
parts = {'fail-to-pass tests': M.caught_grader.mean(), '+ pass-to-pass tests': (M.caught_harness - M.caught_grader).mean(),
         '+ other existing tests': (M.caught_any - M.caught_harness).mean(), 'no existing test': 1 - M.caught_any.mean()}
NUM['blindspot'] = dict(tasks=len(M), planted=int(M.n_mut.sum()), **{k: float(v) for k, v in parts.items()},
                        accepted_by_harness=float(1 - M.caught_harness.mean()), accepted_ci=boot(M, lambda q: 1 - q.caught_harness.mean(), 'repo'),
                        f2p_ci=boot(M, lambda q: q.caught_grader.mean(), 'repo'))
from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(W, 1.25 if PAPER else 2.0)); left = 0; cols = [BLUE, AQUA, YELLOW, GREY]
for (k, v), c in zip(parts.items(), cols):
    ax.barh(0, v, left=left, color=c, height=0.5, edgecolor=SURF, linewidth=2)
    if v >= 0.08: ax.text(left + v / 2, 0, f'{v:.0%}', ha='center', va='center', color='white' if c == BLUE else INK, fontsize=FS, fontweight='bold')
    else: ax.text(left + v / 2, 0.3, f'{v:.0%}', ha='center', va='bottom', color=INK, fontsize=FS, fontweight='bold')
    left += v
ax.set_xlim(0, 1); ax.set_ylim(-0.4, 0.55); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.legend(handles=[Patch(color=c, label=k) for k, c in zip(parts, cols)], frameon=False, fontsize=6.5 if PAPER else 8.5, ncol=2 if PAPER else 4, loc='upper center', bbox_to_anchor=(0.5, -0.5),
          handlelength=1.2, columnspacing=1.0)
fig.suptitle(f'Who catches a bug planted at the fix site? ({int(M.n_mut.sum())} bugs, {len(M)} SWE-rebench tasks)', x=0.01, ha='left', fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(FIG / 'fig2_blind_spot.png', dpi=300, bbox_inches='tight', pad_inches=0.03); plt.close(fig)

# ------------------------------------------------------------ E1: Gemma localization
_loc = g('gemma_out/loc_scores.csv'); L = pd.read_csv(_loc[0] if _loc else RES / 'e1_gemma_localization.csv')
names = {'bm25': 'issue text only', 'cov': 'failing-test coverage', 'pv1': 'provided graph (+1 hop)', 'rt1': 'runtime traces (+1 hop)'}
order = ['bm25', 'pv1', 'cov', 'rt1']; E1 = {}
for c in order:
    q = L[L.cond == c]; E1[c] = dict(n=len(q), recall=float(q.in_cands.mean()), top1=float(q.top1.mean()),
                                     top1_ci=[float(v) for v in np.percentile([rng.choice(q.top1.values.astype(float), len(q)).mean() for _ in range(4000)], [2.5, 97.5])])
piv = L.pivot_table(index='task', columns='cond', values='top1', aggfunc='first').dropna().astype(bool)
from scipy.stats import binomtest
def paired(a, b):
    w, l = int((piv[a] & ~piv[b]).sum()), int((~piv[a] & piv[b]).sum()); return dict(wins=w, losses=l, p=float(binomtest(w, w + l).pvalue))
NUM['E1'] = dict(conditions=E1, rt1_vs_pv1=paired('rt1', 'pv1'), rt1_vs_bm25=paired('rt1', 'bm25'), rt1_vs_cov=paired('rt1', 'cov'))
fig, ax = plt.subplots(figsize=(W, 2.3 if PAPER else 3.0)); yy = np.arange(len(order))[::-1]
for y_, c in zip(yy, order):
    e = E1[c]; ax.barh(y_ + 0.18, e['recall'], height=0.32, color=GREY, edgecolor=SURF, linewidth=2)
    ax.barh(y_ - 0.18, e['top1'], height=0.32, color=BLUE if c == 'rt1' else '#8fb4e6', edgecolor=SURF, linewidth=2)
    ax.errorbar(e['top1'], y_ - 0.18, xerr=[[e['top1'] - e['top1_ci'][0]], [e['top1_ci'][1] - e['top1']]], color=INK, lw=1, capsize=2)
    ax.text(e['recall'] + 0.01, y_ + 0.18, f"{e['recall']:.0%}", va='center', fontsize=FS2, color=INK2)
    ax.text(e['top1_ci'][1] + 0.015, y_ - 0.18, f"{e['top1']:.0%}", va='center', fontsize=FS, color=INK, fontweight='bold')
ax.set_yticks(yy); ax.set_yticklabels([names[c] for c in order]); ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GREY, label='fix among the 15 candidates'), Patch(color=BLUE, label="Gemma 4's first pick is the fix (95% CI)")],
          frameon=False, fontsize=6.5 if PAPER else 8.5, loc='upper center', bbox_to_anchor=(0.3 if PAPER else 0.35, -0.16), ncol=1 if PAPER else 2)
fig.suptitle('Gemma 4 31B localization (dev set): who chooses the candidates it sees', x=0.01, ha='left', fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(FIG / 'fig3_gemma_localization.png', dpi=300, bbox_inches='tight', pad_inches=0.03); plt.close(fig)

# ------------------------------------------------------------ E2: Gemma repair patches vs pre-existing tests
rows = [json.load(open(f)) for f in g('gemma_eval/*.json')]
if rows: P = pd.DataFrame([dict(task=r['task'], repo=r['repo'], **s) for r in rows if r.get('status') == 'graded' for s in r['samples']])
else:
    P = pd.read_csv(RES / 'e2_gemma_patch_samples.csv')
    for c in ('parsed', 'valid', 'noop', 'f2p', 'harness', 'visible', 'extended', 'timeout'):
        P[c] = P[c].map({True: True, False: False, 'True': True, 'False': False})
P['ok'] = P.valid.fillna(False).astype(bool) & ~P.noop.fillna(False).astype(bool)
P['right'] = P.ok & P.harness.fillna(False).astype(bool); P['vis'] = P.ok & P.visible.fillna(False).astype(bool)
V = P[P.ok]; wrong, right = V[~V.right], V[V.right]; greedy = P[P.i == 0]
sel = []
for tid, q in P.groupby('task'):
    q = q.sort_values('i'); cand = q[q.vis]; pick = cand.iloc[0] if len(cand) else q.iloc[0]; sel.append((bool(q.iloc[0].right), bool(pick.right), bool(q.right.any())))
sel = np.array(sel)
NUM['E2'] = dict(tasks=int(P.task.nunique()), answers=len(P), valid=float(P.valid.fillna(False).mean()), greedy_resolve=float(greedy.right.mean()),
                 sampled_resolve=float(P[P.i > 0].right.mean()), best_of_5=float(P.groupby('task').right.any().mean()),
                 wrong=len(wrong), right=len(right), wrong_flagged=float((~wrong.vis).mean()), right_flagged=float((~right.vis).mean()),
                 p_right_given_flag=float(V[~V.vis].right.mean()), p_right_given_pass=float(V[V.vis].right.mean()),
                 wrong_flagged_ci=boot(wrong.assign(f=~wrong.vis), lambda q: q.f.mean(), 'task'),
                 select_greedy=float(sel[:, 0].mean()), select_visible=float(sel[:, 1].mean()), mixed_tasks=int(((P.groupby('task').right.any()) & ~(P.groupby('task').right.all())).sum()))
fig, ax = plt.subplots(figsize=(W, 1.35 if PAPER else 1.9))
for y_, (lab, q) in enumerate([(f'wrong patches (n={len(wrong)})', wrong), (f'correct patches (n={len(right)})', right)]):
    f = (~q.vis).mean()
    ax.barh(y_, f, color=ORANGE, height=0.5, edgecolor=SURF, linewidth=2); ax.barh(y_, 1 - f, left=f, color=GREY, height=0.5, edgecolor=SURF, linewidth=2)
    ax.text(f + 0.01 if f < 0.08 else f / 2, y_, f'{f:.0%}', ha='left' if f < 0.08 else 'center', va='center', fontsize=FS, fontweight='bold',
            color=INK if f < 0.08 else 'white')
ax.set_yticks([0, 1]); ax.set_yticklabels([f'wrong patches (n={len(wrong)})', f'correct patches (n={len(right)})']); ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.legend(handles=[Patch(color=ORANGE, label='an existing test that executes the edit fails'), Patch(color=GREY, label='all such tests pass')],
          frameon=False, fontsize=6.5 if PAPER else 8.5, loc='upper center', bbox_to_anchor=(0.3 if PAPER else 0.5, -0.32), ncol=1 if PAPER else 2)
fig.suptitle(f"Gemma 4 31B patches against the old tests ({NUM['E2']['tasks']} dev tasks, {len(V)} patches)", x=0.01, ha='left', fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(FIG / 'fig4_gemma_patches.png', dpi=300, bbox_inches='tight', pad_inches=0.03); plt.close(fig)

json.dump(NUM, open(FIG / 'numbers.json', 'w'), indent=1)
print(json.dumps(NUM, indent=1)); print('figures in', FIG)
