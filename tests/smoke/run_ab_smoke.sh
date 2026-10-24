#!/usr/bin/env bash
# End-to-end smoke test of the dev-set A/B machinery without a GPU: a scripted OpenAI-compatible server plays the model
# on the two-task mini fixture; both prompts run for two seeds; grading prints resolve rates and the paired tests.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); W=$(mktemp -d)
cp "$ROOT/propagator/proplib.py" "$ROOT/experiments/06_agent_ab/ab_agent.py" "$ROOT/experiments/06_agent_ab/grade_ab.py" "$W/"
python3 "$ROOT/tests/smoke/drive.py" "$W" "$ROOT/tests/fixtures/mini_competition" "$W" | tail -8
SEED=1 python3 "$ROOT/tests/smoke/drive.py" "$W" "$ROOT/tests/fixtures/mini_competition" "$W" > /dev/null
cd "$W" && PROP_WORKDIR="$W" PROP_DATA="$ROOT/tests/fixtures/mini_competition" PROP_TMP="$W/tmp" python3 grade_ab.py | grep -E "seed|ALL SEEDS|permutation"
echo "A/B smoke test passed ($W)"
