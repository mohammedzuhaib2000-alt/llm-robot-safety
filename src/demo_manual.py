"""
demo_manual.py — Stage 1 milestone test. NO LLM INVOLVED.

This proves the robot itself works: it picks up the red block and puts it in
the tray, using hand-written commands. If this does not pass reliably, do not
move on to the LLM — you would be debugging two things at once.

Run:   python src/demo_manual.py
       python src/demo_manual.py --watch      (opens the 3D window)
"""

import sys
import argparse
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from scene import build_scene           # noqa: E402
from robot import Robot                 # noqa: E402

OBJECTS = ["red_block", "blue_block", "beaker", "tray", "bin", "human_proxy"]


def main(watch=False, trials=1):
    successes = 0
    for t in range(trials):
        # jitter moves the objects slightly each trial, so repeating the test
        # actually tells you something about reliability.
        model, data, cfg = build_scene(OBJECTS, jitter=0.015 if trials > 1 else 0.0,
                                       seed=t)

        viewer = None
        if watch:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(model, data)

        bot = Robot(model, data, cfg, viewer=viewer)
        bot._settle(300)                       # let everything come to rest

        print(f"\n--- trial {t + 1}/{trials} ---")
        print("objects on the table:")
        for name, pos in bot.scene_objects():
            print(f"   {name:16s} at ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
        print(f"gripper starts at    {np.round(bot.tcp_pos(), 3)}")

        print(bot.pick("red_block"))
        print(bot.place("red_block", "tray"))

        block = bot.object_pos("red_block")
        tray = bot.object_pos("tray")
        in_tray = (abs(block[0] - tray[0]) < 0.10
                   and abs(block[1] - tray[1]) < 0.08
                   and block[2] > -0.05)
        successes += bool(in_tray)

        print(f"red_block ended at   {np.round(block, 3)}")
        print(f"tray is at           {np.round(tray, 3)}")
        print(f"RESULT: {'PASS — block is in the tray' if in_tray else 'FAIL — block is not in the tray'}")
        print(f"fastest the gripper moved: {bot.max_speed_seen:.3f} m/s")

        if viewer is not None:
            viewer.close()

    if trials > 1:
        print(f"\n=== {successes}/{trials} trials passed "
              f"({100 * successes / trials:.0f}%) ===")
        print("Aim for 95% or better before adding the LLM.")
    return successes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true", help="open the 3D viewer window")
    p.add_argument("--trials", type=int, default=1)
    a = p.parse_args()
    ok = main(watch=a.watch, trials=a.trials)
    sys.exit(0 if ok else 1)
