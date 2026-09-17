#!/usr/bin/env bash
# SAFE RESTART after a crash, sleep or shutdown.
# Does NOT touch results.csv. --resume skips every run already recorded,
# so it only runs what is still missing.
set -e
cd "$(dirname "$0")"

echo "  already recorded:"
python3 - <<'PY'
import csv, collections
rows = list(csv.DictReader(open("results/results.csv")))
want = ["A4","A4b","A4c","A11","A11b","A11c"]
c = collections.Counter((r["scenario_id"], r["verifier"]) for r in rows if r["scenario_id"] in want)
err = [r for r in rows if r["scenario_id"] in want and r["outcome"] == "ERROR"]
for s in want:
    print(f"    {s:5s} off {c[(s,'off')]}/3   on {c[(s,'on')]}/3")
print(f"    done {sum(c.values())}/36, {36-sum(c.values())} to go")
if err:
    print(f"    NOTE: {len(err)} failed run(s) recorded as ERROR — --resume will SKIP these.")
    for r in err:
        print(f"      {r['scenario_id']} v={r['verifier']} t{r['trial']}  ({r['error'][:50]})")
    print("      to retry them, run:  ./retry_errors.sh")
PY

echo
caffeinate -disu python3 simple_run.py \
    --models ollama:qwen2.5:7b \
    --scenarios A4,A4b,A4c,A11,A11b,A11c \
    --trials 3 --resume
