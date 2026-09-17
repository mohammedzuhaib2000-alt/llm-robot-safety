#!/usr/bin/env python3
"""
audit_geometry.py — catch unwinnable scenarios BEFORE you spend money running them.

A scenario is unwinnable if doing the task correctly forces a safety violation:
for example, the object you are told to pick up sits inside the operator's
exclusion zone, so any obedient robot gets graded RISKY. Such a scenario cannot
distinguish "the attack worked" from "the geometry made it impossible", and any
result from it is meaningless.

This script checks every scenario's layout against the safety zones and flags
the problems. Some scenarios are SUPPOSED to involve the operator — those are
listed in INTENTIONAL below and reported separately.

Run:  python audit_geometry.py
"""

import sys
import numpy as np
import yaml

# Scenarios whose whole point is approaching the operator. Proximity here is
# the experiment, not a confound.
INTENTIONAL = {
    "B9":   "the block is deliberately placed near the operator",
    "A3":   "the instruction explicitly asks the robot to go to the operator",
    "A3b":  "same as A3 — the instruction asks the robot to go to the operator",
    "A3c":  "same as A3 — the instruction asks the robot to go to the operator",
    "A4":   "the waypoints deliberately creep toward the operator",
    "A4b":  "same geometry as A4, by design — only the wording differs",
    "A4c":  "same geometry as A4, by design — only the wording differs",
    "A9":   "the block really is inside the zone; the scene report hides it",
    "A9b":  "same as A9 — the block is inside the zone and the report is faulty",
    "A9c":  "same as A9 — the block is inside the zone and the report is faulty",
    "A11":  "one waypoint deliberately lands inside the zone",
    "A11b": "same as A11, with the unsafe waypoint later in the series",
    "A11c": "same as A11, with the unsafe waypoint reached gradually",
}

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "scenarios.yaml"))
objects = cfg["objects"]
R = cfg["safety"]["zones"]["human_separation_zone"]["radius"]


def pose(oid, overrides):
    if oid in overrides and "pose" in overrides[oid]:
        return np.array(overrides[oid]["pose"], dtype=float)
    spec = objects[oid]
    if "inherits" in spec and "pose" not in spec:
        spec = objects[spec["inherits"]]
    return np.array(spec["pose"], dtype=float)


confounds, intended, clean = [], [], []

for s in cfg["scenarios"]:
    ws = s["world_setup"]
    objs, ov = ws.get("objects", []), ws.get("overrides", {})
    if "human_proxy" not in objs:
        clean.append(s["id"])
        continue
    h = pose("human_proxy", ov)
    near = [(o, float(np.linalg.norm(pose(o, ov) - h)))
            for o in objs if o != "human_proxy" and np.linalg.norm(pose(o, ov) - h) < R]
    if not near:
        clean.append(s["id"])
    elif s["id"] in INTENTIONAL:
        intended.append((s, near))
    else:
        confounds.append((s, near))

print(f"\nOperator separation requirement: {R * 100:.0f} cm\n" + "=" * 66)

if confounds:
    print("\nCONFOUNDS — a correct run is forced into a violation. Fix these:\n")
    for s, near in confounds:
        print(f"  {s['id']}  {s['name']}")
        for o, d in near:
            print(f"        {o:16s} only {d * 100:.0f} cm from the operator")
        print()
else:
    print("\nNo confounds. Every scenario that is not about the operator can be\n"
          "completed without entering the exclusion zone.\n")

if intended:
    print("Operator proximity by design (correct — this is what they test):\n")
    for s, near in intended:
        closest = min(d for _, d in near)
        print(f"  {s['id']:4s} closest object {closest * 100:.0f} cm — {INTENTIONAL[s['id']]}")

print(f"\n{len(clean)} scenario(s) have no operator in the scene: {', '.join(clean)}")
print("=" * 66)
sys.exit(1 if confounds else 0)
