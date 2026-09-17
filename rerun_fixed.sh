#!/usr/bin/env bash
# Re-run the scenarios invalidated by the `done` truncation defect.
# Run this ONLY after the previous batch has stopped.
set -e
cd "$(dirname "$0")"

STAMP=$(date +%Y%m%d-%H%M%S)
cp results/results.csv "results/results_prefix_${STAMP}.csv"

# Quarantine every row produced while `done` could truncate a conversation.
# A4/A11 families will be re-run. A10 family is discarded (C4 dropped).
python3 - <<'PY'
import csv, pathlib
BAD = {"A4","A4b","A4c","A10","A10b","A10c","A11","A11b","A11c"}
src = pathlib.Path("results/results.csv")
rows = list(csv.DictReader(open(src)))
import simple_run
keep = [r for r in rows if r["scenario_id"] not in BAD]
drop = [r for r in rows if r["scenario_id"] in BAD]
with open("results/results_discarded_donebug.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(drop)
with open(src,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=simple_run.FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in keep:
        w.writerow({k: r.get(k,"") for k in simple_run.FIELDS})
print(f"  kept {len(keep)} valid rows, quarantined {len(drop)} into results/results_discarded_donebug.csv")
PY

echo "  starting 36 runs (A4/A4b/A4c/A11/A11b/A11c x off,on x 3 trials)"
caffeinate -i python3 simple_run.py \
    --models ollama:qwen2.5:7b \
    --scenarios A4,A4b,A4c,A11,A11b,A11c \
    --trials 3 --resume 2>&1 | tee "results/rerun_${STAMP}.log"
