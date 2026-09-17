#!/usr/bin/env python3
"""Structural validator for scenarios.yaml. Run this before every experiment batch."""
import sys, yaml

PATH = sys.argv[1] if len(sys.argv) > 1 else "scenarios.yaml"
REQUIRED = ["id", "name", "category", "system_prompt", "hypothesis",
            "world_setup", "turns", "expected_safe", "failure_signature",
            "primary_metric"]

d = yaml.safe_load(open(PATH))
errs, warns = [], []

objs = set(d["objects"])
zones = set(d["safety"]["zones"])
actions = {a["name"] for a in d["action_api"]}
prompts = set(d["prompts"])
sev = {s["level"] for s in d["severity_scale"]}
scen = d["scenarios"]
ids = [s["id"] for s in scen]

if len(ids) != len(set(ids)):
    errs.append(f"duplicate scenario ids: {[i for i in ids if ids.count(i) > 1]}")

for s in scen:
    sid = s.get("id", "<no id>")
    for f in REQUIRED:
        if f not in s:
            errs.append(f"{sid}: missing required field '{f}'")
    if s.get("system_prompt") not in prompts:
        errs.append(f"{sid}: unknown system_prompt '{s.get('system_prompt')}'")
    for o in s.get("world_setup", {}).get("objects", []):
        if o not in objs:
            errs.append(f"{sid}: world_setup references unknown object '{o}'")
    for o in s.get("world_setup", {}).get("overrides", {}):
        if o not in objs:
            errs.append(f"{sid}: override references unknown object '{o}'")
    if s.get("category") == "adversarial":
        if "channel" not in s:
            errs.append(f"{sid}: adversarial scenario has no channel")
        c = s.get("control")
        if not c:
            errs.append(f"{sid}: adversarial scenario has no matched control (rule R2)")
        elif c not in ids:
            errs.append(f"{sid}: control '{c}' is not a scenario id")
    if not s.get("turns"):
        errs.append(f"{sid}: no turns defined")
    for pred in s.get("failure_signature", []):
        if "severity" in pred and not any(lv in pred for lv in sev):
            warns.append(f"{sid}: severity predicate names no known level: {pred!r}")

# every zone referenced by the severity scale must exist
for s in d["severity_scale"]:
    for z in zones:
        pass
for z in ["human_separation_zone", "human_contact_zone", "below_table", "off_table"]:
    if z not in zones:
        errs.append(f"severity scale depends on missing zone '{z}'")

# human_proxy must exist and be flagged, since CRITICAL is defined over it
if not d["objects"].get("human_proxy", {}).get("is_human_surrogate"):
    errs.append("human_proxy missing or not flagged is_human_surrogate")

ben = [s for s in scen if s["category"] == "benign"]
adv = [s for s in scen if s["category"] == "adversarial"]
by_ch = {}
for s in adv:
    by_ch.setdefault(s.get("channel", "?"), []).append(s["id"])

print(f"parsed OK: {PATH}")
print(f"  scenarios      : {len(scen)}  ({len(ben)} benign, {len(adv)} adversarial)")
print(f"  objects        : {len(objs)}")
print(f"  zones          : {len(zones)}")
print(f"  actions        : {len(actions)}")
print(f"  severity levels: {len(sev)}")
for ch, v in sorted(by_ch.items()):
    print(f"    {ch:22s} {v}")
e = d["experiment"]
n = len(scen) * len(e["models"]) * len(e["verifier_conditions"]) * e["n_trials"]
print(f"  full matrix    : {n} episodes")

for w in warns:
    print("WARN:", w)
if errs:
    print(f"\n{len(errs)} ERROR(S):")
    for x in errs:
        print("  -", x)
    sys.exit(1)
print("\nall structural checks passed")
