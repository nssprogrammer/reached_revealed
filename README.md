# Reached but Not Revealed

Code, notebooks and derived data for the paper **"Reached but Not Revealed: How Much Tests See at the Fix Site, and What It Means for Gemma 4 Coding Agents"** (Kaggle Gemma 4 Developer Agent competition, paper track). The PDF and LaTeX sources are in [`paper/`](paper/).

The pipeline traces, for every test, which functions run, then measures:

- whether pre-existing tests that **reach** a bug also **reveal** it;
- which test sets catch bugs planted at the **fix site** (the functions a reference fix edits);
- what context helps Gemma 4 31B localize a fix, and how well old tests judge its patches;
- whether three agent habits derived from these results improve a competition agent (a paired dev-set A/B).

Everything except Gemma inference runs on Kaggle CPUs. Gemma runs use the competition checkpoint `gemma-4-31b-it-qat-w4a16-ct` with vLLM 0.19.1 on Kaggle's L4 ×4 GPUs, with internet off.

## Repository layout

```
propagator/                 libraries imported by the experiments
  proplib.py                competition environments, tracer, real-bug response (Step 8)
  proplib9.py               planted bugs at the fix site and its neighbours (Step 9)
  genlib.py                 the same measurements on eight pip-installable libraries
  swelib.py                 the same measurements on SWE-rebench tasks (uv environments, Python 3.9-3.13)
  probe.py                  the 2.7 KB stdlib tracer the agent writes (no curly braces: it is embedded in an ADK instruction)
experiments/
  01_competition_trace/     Step 8 (real bugs) and Step 9 (planted bugs) runners for the competition dev set
  02_competition_analysis/  Steps 10-13: predictors and amplification, localization, provided graph vs runtime graph, graders
  03_libraries/             runner for the eight libraries
  04_swe_rebench/           runner and report for 150 SWE-rebench candidates (104 reproduced, 96 repositories)
  05_gemma/                 E0 prompt pack, E1/E2 generation with Gemma 4 31B, E2 patch grading
  06_agent_ab/              agent loop with the competition's tools and limits, A/B runner, grading with McNemar and a permutation test
  07_figures/               Figures 1-4 and numbers.json (works offline from results/)
notebooks/                  the same code as Kaggle notebooks (each cell writes the library it needs with %%writefile)
agent/                      competition agent prompts (v2 as scored, v3.2, v3.3b) and the submission notebooks
paper/                      paper.tex, IEEEtran.cls, figures, compiled PDF
results/                    derived tables behind the paper's numbers (see results/README.md)
tests/                      fixtures (a two-task mini repository) and smoke tests that need no GPU and no competition data
```

## Paper claims and the code that produces them

| Paper claim | Value | Produced by | Inputs |
|---|---|---|---|
| Competition dev set: old tests run the buggy function in 94/101 tasks (median 242 tests), silent in 93/94 | Sec. IV | `experiments/01_competition_trace/run_step8_real_bugs.py` (`proplib.run_step8`, `report8`) | competition data |
| 1,422 planted bugs; 16% of covered mutants make no test fail | Sec. IV | `experiments/01_competition_trace/run_step9_planted_bugs.py` (`proplib9`) | Step 8 output |
| Transient amplification median 5.1× (max 28×); coverage predicts blast radius, graph measures add nothing | Sec. IV | `02_competition_analysis/step10_predictors_amplification.py`, `step10b_coverage_free_and_depth.py` | Step 9 output |
| Ochiai top-5 30% → one hop 53% | Sec. IV | `02_competition_analysis/step11_observability_localization.py`, `step12_localization_shape.py` | Step 9 output + competition data |
| Runtime vs provided graph 52% vs 38% top-5 (p = 1.4e-6); F2P tests catch 48% of fix-site bugs vs 81% for any covering test | Sec. IV | `02_competition_analysis/step13_provided_graph_and_graders.py` | Step 9 output + competition data (graphs) |
| Eight libraries: amplification median 3.7× (2.0-14.7×), 339 planted bugs, no depth effect within mutants (p = 0.66) | Sec. IV | `03_libraries/run_libraries.py` (`genlib`) | internet (PyPI, GitHub) |
| SWE-rebench: 104 of 150 reproduce (96 repos); 77/78 silent (95% CI 96-100%), median 27 old tests | Fig. 1 | `04_swe_rebench/run_swe_rebench.py` (`swelib`), then `07_figures/figures.py` | internet (Hugging Face, GitHub, PyPI) |
| Fix-site breakdown 55% / +21% / +6% / 18%; full harness accepts 24% (18-31%) | Fig. 2 | same as above | SWE-rebench output |
| E1: Gemma's first pick correct 64% (runtime traces) vs 53% (provided graph), p = 0.02; 43% issue text only | Fig. 3 | `05_gemma/e0_prompt_pack.py` → `e1_e2_generate_gemma.py` → `07_figures/figures.py` | Step 9 output, competition data, model |
| E2: old tests flag 43% of wrong Gemma patches and 2% of correct ones; a flag means wrong 96% of the time | Fig. 4 | `05_gemma/e2_grade_patches.py` → `07_figures/figures.py` | E1/E2 output |
| A/B: v3.2 vs v2 resolves 15.6% vs 10.4% of 212 task runs (+5.2 points, 95% CI 1.4-9.4), 14 tasks better vs 4, permutation p = 0.026 | Sec. VI | `06_agent_ab/run_ab.py` (×2 seeds) → `06_agent_ab/grade_ab.py` | competition data, model, wheelhouse |

`experiments/07_figures/figures.py` regenerates Figures 1-4 and `numbers.json` from the tables in `results/`, with no other inputs:

```bash
pip install numpy pandas scipy matplotlib
FIG_DIR=out/figures python experiments/07_figures/figures.py     # PAPER=1 for column-width figures
```

The values match `results/numbers.json` (up to floating-point rounding; bootstrap intervals use fixed seeds).

## Reproducing on Kaggle

Each notebook in `notebooks/` is self-contained: its first cells write the libraries it needs into `/kaggle/working`. Run them in this order.

| Order | Notebook | Hardware | Internet | Inputs | Time |
|---|---|---|---|---|---|
| 1 | `propagator_end_to_end.ipynb` (Steps 8-9) | CPU | off | competition data | ~4-5 h |
| 2 | `propagator_step10.ipynb` … `propagator_step13.ipynb` | CPU | off | output of 1 (+ competition data for 11-13) | minutes each |
| 3 | `propagator_generalization.ipynb` (eight libraries) | CPU | **on** | none | ~1-2 h |
| 4 | `propagator_swe_rebench.ipynb` | CPU | **on** | none (optionally an earlier `swe_out.tgz` to resume) | ~7 h |
| 5 | `propagator_gemma_pack.ipynb` (E0) | CPU | off | competition data, output of 1 | ~6 min |
| 6 | `propagator_gemma_run.ipynb` (E1, E2 generation) | **L4 ×4** | off | output of 5, model `google/gemma-4` → `gemma-4-31b-it-qat-w4a16-ct`, dataset `metric/gemma-4-developer-agent-wheelhouse` | ~1 h |
| 7 | `propagator_gemma_eval.ipynb` (E2 grading) | CPU | off | competition data, outputs of 1, 5, 6 | ~45 min |
| 8 | `propagator_ab_run.ipynb` (seed 0, then seed 1 with `SEEDS = [1]` and the seed-0 `ab_out.tgz` attached) | **L4 ×4** | off | competition data, model, wheelhouse, output of 1 | ~2.3 h per seed |
| 9 | `propagator_ab_grade.ipynb` | CPU | off | competition data, `ab_out.tgz` with both seeds | ~25 min |
| 10 | `propagator_figures.ipynb` | CPU | off | `swe_out.tgz`, `gemma_out.tgz`, `gemma_eval.tar.gz` (or nothing: falls back to `results/`) | seconds |

Running the scripts outside Kaggle: put `propagator/` on the path (`export PYTHONPATH=$PWD/propagator`) and point the environment variables at your folders: `PROP_WORKDIR` (outputs), `PROP_DATA` (competition data with `snapshots/`, `wheels/`, `graphs/`), `PROP_TMP` (scratch).

## Tests (no GPU, no competition data)

```bash
pip install pytest numpy pandas scipy packaging
python tests/smoke/test_probe.py          # the agent's tracer: snippet localization and --focus regression check
bash tests/smoke/run_ab_smoke.sh          # the A/B agent loop against a scripted model server, two seeds, grading and paired tests
```

## Notes and limitations

- Python and pytest only; the tracer sees Python-level calls.
- The A/B uses a re-implementation of the competition's tool interface, not the official harness, with thinking off in both arms. Absolute resolve rates are therefore not leaderboard rates; the comparison between prompts is controlled.
- Kaggle paths are the defaults (`/kaggle/working`, `/kaggle/input`); every script accepts the environment overrides above.
- The agent prompt v2 builds on matterhorn3838's "Gemma 4 SuperAgent" notebook and the official sample submission; see `agent/README.md`.

## Citation

```
[Author Name]. Reached but Not Revealed: How Much Tests See at the Fix Site, and What It Means for
Gemma 4 Coding Agents. Kaggle Gemma 4 Developer Agent Competition, paper track, 2026.
```
