#!/usr/bin/env bash
# Clear every timed-out row and re-attempt only the missing cells.
# Safe to run in the background while you write. Ctrl-C any time.
set -e
cd "$(dirname "$0")"
python3 - <<'PY'
import csv, shutil, simple_run
shutil.copy("results/results.csv", "results/results_before_gapfill.csv")
rows = list(csv.DictReader(open("results/results.csv")))
keep = [r for r in rows if r["outcome"] != "ERROR"]
with open("results/results.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=simple_run.FIELDS, extrasaction="ignore")
    w.writeheader(); w.writerows(keep)
print(f"  cleared {len(rows)-len(keep)} timed-out rows (backup: results/results_before_gapfill.csv)")
PY
echo "  --- adversarial gaps (3 trials) ---"
caffeinate -disu python3 simple_run.py --models ollama:qwen2.5:7b \
    --scenarios A7,A9,A11b --trials 3 --resume
echo "  --- benign gaps (10 trials) ---"
caffeinate -disu python3 simple_run.py --models ollama:qwen2.5:7b \
    --scenarios B8,B9,B10 --trials 10 --resume
