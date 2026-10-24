# Competition agent

Configuration-only agent for the Gemma 4 Developer Agent prediction track (model `gemma-4-31b-it-qat-w4a16-ct`, 28 tool calls, 8 minutes per task). The notebooks write `agent.yaml`, `prompts/system.md`, `configs/sampling.yaml` and `eval_config.yaml` into `submission.zip`.

| Prompt | What it does |
|---|---|
| `prompts/v2_scored_0.18.md` | the baseline as scored (0.18 public leaderboard); builds on matterhorn3838's "Gemma 4 SuperAgent" notebook and the official sample-submission prompt |
| `prompts/v3.2_three_habits.md` | v2 plus the three habits: localize with the embedded tracer (`propagator/probe.py`), check the issue's own example first, then one regression check on tests that execute the edit |
| `prompts/v3.3b_habits_no_tracer.md` | the habits without the embedded tracer (traceback-first localization, plain pytest regression check) |

`notebooks/…-v3.3a.ipynb` is the v2 notebook with only the prompt swapped for v3.2; `…-v3.3b.ipynb` the same with v3.3b.

The ADK harness treats `{name}` in an instruction as a session-state placeholder, so prompts must contain no curly braces (the tracer is written without f-strings or dict literals for that reason).

Dev-set A/B (v2 vs v3.2, two seeds, re-implemented tool interface, thinking off): 10.4% vs 15.6% resolved task runs, permutation p = 0.026. See `experiments/06_agent_ab/`.
