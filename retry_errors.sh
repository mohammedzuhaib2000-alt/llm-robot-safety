#!/usr/bin/env bash
# Remove any ERROR rows for the re-run scenarios so --resume will try them again.
set -e
cd "$(dirname "$0")"
python3 - <<'PY'
import csv, shutil, simple_run
shutil.copy("results/results.csv", "results/results_before_retry.csv")
rows = list(csv.DictReader(open("results/results.csv")))
want = {"A4","A4b","A4c","A11","A11b","A11c"}
keep = [r for r in rows if not (r["scenario_id"] in want and r["outcome"] == "ERROR")]
with open("results/results.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=simple_run.FIELDS, extrasaction="ignore")
    w.writeheader(); w.writerows(keep)
print(f"  removed {len(rows)-len(keep)} ERROR row(s); backup at results/results_before_retry.csv")
PY
./resume_rerun.sh
