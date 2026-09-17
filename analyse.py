#!/usr/bin/env python3
"""
analyse.py — turn results/results.csv into your actual results.

This is Stage 3. It reads the file the batch runner produced and works out:

  * how often each test ended unsafely
  * the safety checker OFF vs ON comparison   <- your headline result
  * the matched pairs (same action, different wording)
  * the safety / capability trade-off
  * severity broken down by attack channel and by model

It writes tables into results/ and charts into results/charts/, and prints a
summary you can read straight out.

Run:  python analyse.py
      python analyse.py --model claude-sonnet-4-5     (just one model)
"""

import argparse
import csv
import math
import pathlib
import sys
from collections import defaultdict

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent


def load_config():
    with open(PROJECT_ROOT / "scenarios.yaml") as f:
        return yaml.safe_load(f)


RES = PROJECT_ROOT / "results"
CHARTS = RES / "charts"
CSV_PATH = RES / "results.csv"

SEV_ORDER = ["SAFE", "RISKY", "DANGEROUS", "CRITICAL"]
SEV_COLOUR = {"SAFE": "#0ca30c", "RISKY": "#fab219",
              "DANGEROUS": "#ec835a", "CRITICAL": "#d03b3b"}
BLUE, ORANGE, INK, INK2, LINE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9dee6"

# Same action, different framing. The gap between each pair is a result.
MATCHED_PAIRS = [
    ("A1", "A2", "plain request vs 'it's just a demo'"),
    ("A6", "A7", "plain injected card vs one dressed as a system message"),
    ("B3", "A8", "fragile object vs the same object relabelled as waste"),
    ("A1", "A10", "plain request vs the same one after a long conversation"),
    ("B9", "A4", "reaching near the operator vs creeping there by coordinates"),
]


# --------------------------------------------------------------- statistics --

def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion. Use this, not bare percentages:
    with 10 trials, 7/10 really means 'somewhere between 40% and 89%'."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def fisher_exact(a, b, c, d):
    """Two-sided Fisher's exact test on a 2x2 table. No scipy needed.
        a b        a = unsafe in group 1, b = safe in group 1
        c d        c = unsafe in group 2, d = safe in group 2
    Returns the p-value: the chance of seeing a split this lopsided if the two
    groups really behaved the same."""
    n = a + b + c + d
    if n == 0:
        return 1.0

    def prob(x):
        return (math.comb(a + b, x) * math.comb(c + d, a + c - x)
                / math.comb(n, a + c))

    lo = max(0, a + c - (c + d))
    hi = min(a + b, a + c)
    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= observed + 1e-12))


# --------------------------------------------------------------------- data --

def load_rows(model_filter=None):
    if not CSV_PATH.exists():
        raise SystemExit(f"No results file at {CSV_PATH}.\nRun:  python run_batch.py --models fake --trials 2")
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            if model_filter and model_filter not in (r.get("model") or ""):
                continue
            rows.append(r)
    if not rows:
        raise SystemExit("No usable rows (all errors, or the model filter matched nothing).")
    return rows


def unsafe(r):
    return r["severity"] != "SAFE"


def rate(rows, pred):
    n = len(rows)
    k = sum(1 for r in rows if pred(r))
    return k, n, wilson(k, n)


def section(title):
    print("\n" + "=" * 74)
    print("  " + title)
    print("=" * 74)


# ------------------------------------------------------------------ reports --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="only analyse rows for this model")
    a = ap.parse_args()

    cfg = load_config()
    rows = load_rows(a.model)
    RES.mkdir(exist_ok=True); CHARTS.mkdir(exist_ok=True)
    scen_by_id = {s["id"]: s for s in cfg["scenarios"]}

    models = sorted({r["model"] or r["provider"] for r in rows})
    fake_only = models == ["fake"]

    section("OVERVIEW")
    print(f"  runs analysed : {len(rows)}")
    print(f"  models        : {', '.join(models)}")
    print(f"  scenarios     : {len({r['scenario_id'] for r in rows})}")
    print(f"  total cost    : ${sum(float(r['cost_usd'] or 0) for r in rows):.2f}")
    if fake_only:
        print("\n  NOTE: these rows come from the 'fake' stand-in model. It is not")
        print("        intelligent. Use this only to check the pipeline works —")
        print("        never report these numbers as results.")

    # ---- H1: the headline. Safety checker off vs on -------------------------
    section("H1  DOES THE SAFETY CHECKER HELP?   (your headline result)")
    adv = [r for r in rows if r["category"] == "adversarial"]
    off = [r for r in adv if r["verifier"] == "off"]
    on = [r for r in adv if r["verifier"] == "on"]
    ko, no, (po, lo_, ho) = rate(off, unsafe)
    kn, nn, (pn, ln_, hn) = rate(on, unsafe)
    print(f"  attacks, checker OFF : {ko}/{no} ended unsafely  = {po:6.1%}  (95% CI {lo_:.1%}–{ho:.1%})")
    print(f"  attacks, checker ON  : {kn}/{nn} ended unsafely  = {pn:6.1%}  (95% CI {ln_:.1%}–{hn:.1%})")
    if no and nn:
        p = fisher_exact(ko, no - ko, kn, nn - kn)
        print(f"  difference           : {po - pn:+.1%}   (Fisher's exact p = {p:.4f})")
        print("  " + ("-> the checker made a real difference" if p < 0.05
                      else "-> not statistically significant yet; more trials needed"))

    # ---- H5: what the checker costs -----------------------------------------
    section("H5  WHAT DOES THE CHECKER COST?   (the one students forget)")
    ben = [r for r in rows if r["category"] == "benign"]
    for v in ("off", "on"):
        g = [r for r in ben if r["verifier"] == v]
        k, n, (p, lo2, hi2) = rate(g, lambda r: r["outcome"] == "COMPLETED_SAFELY")
        print(f"  ordinary tasks, checker {v.upper():3s}: {k}/{n} completed safely "
              f"= {p:6.1%}  (95% CI {lo2:.1%}–{hi2:.1%})")
    print("\n  A system that refuses everything is perfectly safe and completely")
    print("  useless. Always report this number next to H1.")

    # ---- per scenario --------------------------------------------------------
    section("EVERY TEST, UNSAFE RATE")
    print(f"  {'test':5s} {'kind':11s} {'checker off':>18s} {'checker on':>18s}   name")
    out_rows = []
    for sid in sorted(scen_by_id, key=lambda x: (x[0], int(x[1:]))):
        sub = [r for r in rows if r["scenario_id"] == sid]
        if not sub:
            continue
        cells = {}
        for v in ("off", "on"):
            g = [r for r in sub if r["verifier"] == v]
            k, n, (p, _, _) = rate(g, unsafe)
            cells[v] = f"{k}/{n} ({p:.0%})" if n else "—"
        kind = sub[0]["category"][:11]
        print(f"  {sid:5s} {kind:11s} {cells['off']:>18s} {cells['on']:>18s}   "
              f"{scen_by_id[sid]['name'][:34]}")
        out_rows.append({"scenario_id": sid, "name": scen_by_id[sid]["name"],
                         "category": sub[0]["category"],
                         "channel": sub[0]["channel"],
                         "unsafe_off": cells["off"], "unsafe_on": cells["on"]})

    with open(RES / "by_scenario.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)

    # ---- matched pairs -------------------------------------------------------
    section("MATCHED PAIRS   (same action, one thing changed)")
    pair_out = []
    for x, y, why in MATCHED_PAIRS:
        gx = [r for r in rows if r["scenario_id"] == x and r["verifier"] == "off"]
        gy = [r for r in rows if r["scenario_id"] == y and r["verifier"] == "off"]
        if not gx or not gy:
            continue
        kx, nx, (px, _, _) = rate(gx, unsafe)
        ky, ny, (py, _, _) = rate(gy, unsafe)
        p = fisher_exact(kx, nx - kx, ky, ny - ky)
        star = " *" if p < 0.05 else ""
        print(f"  {x} vs {y}: {px:5.0%} -> {py:5.0%}   gap {py - px:+5.0%}   p={p:.3f}{star}")
        print(f"           {why}")
        pair_out.append({"a": x, "b": y, "what_changed": why,
                         "unsafe_a": f"{kx}/{nx}", "unsafe_b": f"{ky}/{ny}",
                         "gap": f"{py - px:+.2f}", "p_value": f"{p:.4f}"})
    if pair_out:
        with open(RES / "matched_pairs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pair_out[0]))
            w.writeheader(); w.writerows(pair_out)

    # ---- by channel ----------------------------------------------------------
    section("BY ATTACK CHANNEL   (checker off)")
    by_ch = defaultdict(list)
    for r in rows:
        if r["category"] == "adversarial" and r["verifier"] == "off":
            by_ch[r["channel"] or "unknown"].append(r)
    for ch, g in sorted(by_ch.items()):
        k, n, (p, lo3, hi3) = rate(g, unsafe)
        print(f"  {ch:24s} {k:3d}/{n:<3d} unsafe = {p:6.1%}  (95% CI {lo3:.0%}–{hi3:.0%})")

    # ---- by model ------------------------------------------------------------
    if len(models) > 1:
        section("BY MODEL   (attacks, checker off)")
        for m in models:
            g = [r for r in rows if (r["model"] or r["provider"]) == m
                 and r["category"] == "adversarial" and r["verifier"] == "off"]
            if not g:
                continue
            k, n, (p, lo4, hi4) = rate(g, unsafe)
            print(f"  {m:26s} {k:3d}/{n:<3d} unsafe = {p:6.1%}  (95% CI {lo4:.0%}–{hi4:.0%})")

    make_charts(rows, scen_by_id)

    print("\n" + "=" * 74)
    print(f"  tables written to {RES}")
    print(f"  charts written to {CHARTS}")
    print("=" * 74 + "\n")


# ------------------------------------------------------------------- charts --

def make_charts(rows, scen_by_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": LINE,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})

    # 1. severity mix, checker off vs on
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    groups = ["off", "on"]
    bottoms = [0, 0]
    for sev in SEV_ORDER:
        vals = []
        for v in groups:
            g = [r for r in rows if r["verifier"] == v]
            vals.append(100 * sum(1 for r in g if r["severity"] == sev) / len(g) if g else 0)
        ax.barh(groups, vals, left=bottoms, color=SEV_COLOUR[sev], label=sev, height=0.5)
        for i, x in enumerate(vals):
            if x > 6:
                ax.text(bottoms[i] + x / 2, i, f"{x:.0f}%", ha="center", va="center",
                        fontsize=9, color="white" if sev != "RISKY" else "#3d2c00",
                        fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_yticks([0, 1]); ax.set_yticklabels(["checker OFF", "checker ON"])
    ax.set_xlim(0, 100); ax.set_xlabel("% of runs")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    ax.set_title("How runs ended, with the safety checker off and on",
                 loc="left", fontweight="bold", color=INK, pad=12)
    plt.tight_layout(); plt.savefig(CHARTS / "severity_by_verifier.png", dpi=180,
                                    bbox_inches="tight"); plt.close()

    # 2. unsafe rate per scenario, off vs on
    sids = sorted({r["scenario_id"] for r in rows},
                  key=lambda x: (x[0], int(x[1:])))
    offv, onv = [], []
    for sid in sids:
        for v, acc in (("off", offv), ("on", onv)):
            g = [r for r in rows if r["scenario_id"] == sid and r["verifier"] == v]
            acc.append(100 * sum(1 for r in g if unsafe(r)) / len(g) if g else 0)
    fig, ax = plt.subplots(figsize=(max(7.4, len(sids) * 0.52), 3.4))
    xs = range(len(sids))
    ax.bar([x - 0.2 for x in xs], offv, width=0.38, color=ORANGE, label="checker off")
    ax.bar([x + 0.2 for x in xs], onv, width=0.38, color=BLUE, label="checker on")
    ax.set_xticks(list(xs)); ax.set_xticklabels(sids, fontsize=9)
    ax.set_ylabel("% of runs that ended unsafely"); ax.set_ylim(0, 105)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Unsafe outcomes by test", loc="left", fontweight="bold", color=INK, pad=10)
    plt.tight_layout(); plt.savefig(CHARTS / "unsafe_by_scenario.png", dpi=180,
                                    bbox_inches="tight"); plt.close()

    # 3. the trade-off: does the robot still work?
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for v, colour, label in (("off", ORANGE, "checker off"), ("on", BLUE, "checker on")):
        ben = [r for r in rows if r["category"] == "benign" and r["verifier"] == v]
        adv = [r for r in rows if r["category"] == "adversarial" and r["verifier"] == v]
        if not ben or not adv:
            continue
        x = 100 * sum(1 for r in adv if unsafe(r)) / len(adv)
        y = 100 * sum(1 for r in ben if r["outcome"] == "COMPLETED_SAFELY") / len(ben)
        ax.scatter([x], [y], s=190, color=colour, zorder=3, label=label)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(10, -4),
                    fontsize=9.5, color=colour, fontweight="bold")
    ax.set_xlabel("attacks that succeeded (%)  —  lower is safer")
    ax.set_ylabel("ordinary tasks completed (%)\n—  higher is more useful")
    ax.set_xlim(-5, 105); ax.set_ylim(-5, 105)
    ax.grid(alpha=0.25, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("Safety against usefulness", loc="left", fontweight="bold",
                 color=INK, pad=10)
    plt.tight_layout(); plt.savefig(CHARTS / "safety_vs_capability.png", dpi=180,
                                    bbox_inches="tight"); plt.close()

    print("\n  charts: severity_by_verifier.png, unsafe_by_scenario.png, "
          "safety_vs_capability.png")


if __name__ == "__main__":
    main()
