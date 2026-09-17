"""
check_setup.py — run this FIRST.

It checks every part of your setup one at a time and tells you exactly what is
missing and how to fix it. It does not need an API key and does not cost
anything.

Run:  python src/check_setup.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

OK, BAD = "  [OK]   ", "  [FAIL] "
problems = []


def check(label, fn, fix):
    try:
        detail = fn()
        print(f"{OK}{label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as e:
        print(f"{BAD}{label}")
        print(f"         problem: {type(e).__name__}: {e}")
        print(f"         fix:     {fix}")
        problems.append(label)
        return False


print("\nChecking your setup\n" + "-" * 60)

check("Python is 3.10 or newer",
      lambda: (_ for _ in ()).throw(RuntimeError(f"you have {sys.version.split()[0]}"))
      if sys.version_info < (3, 10) else f"{sys.version.split()[0]}",
      "install a newer Python (python.org, or 'brew install python@3.12')")

check("numpy is installed", lambda: __import__("numpy").__version__,
      "pip install numpy")

check("pyyaml is installed", lambda: __import__("yaml").__version__,
      "pip install pyyaml")

check("mujoco is installed", lambda: __import__("mujoco").__version__,
      "pip install mujoco")

if not problems:
    from scene import find_menagerie, load_config, build_scene, SCENARIOS_FILE

    check("scenarios.yaml is present and readable",
          lambda: f"{len(load_config()['scenarios'])} scenarios",
          f"the file should be at {SCENARIOS_FILE}")

    check("the robot model folder (mujoco_menagerie) was found",
          lambda: str(find_menagerie()),
          "git clone https://github.com/google-deepmind/mujoco_menagerie.git")

if not problems:
    check("the world builds and the robot loads",
          lambda: (lambda mdc: f"{mdc[0].nbody} bodies, {mdc[0].nu} actuators")(
              build_scene(["red_block", "tray", "human_proxy"])),
          "check the error above — usually a missing mujoco_menagerie folder")

    def _sim_runs():
        import mujoco
        from robot import Robot
        m, d, c = build_scene(["red_block", "tray"])
        b = Robot(m, d, c)
        b._step(200)
        return f"gripper is at {b.tcp_pos().round(3)}"

    check("the simulation runs and the arm reports its position", _sim_runs,
          "check the error above")

print("-" * 60)
if problems:
    print(f"\n{len(problems)} thing(s) need fixing: {', '.join(problems)}")
    print("Fix them top to bottom, then run this again.\n")
    sys.exit(1)

print("""
Everything works. Next steps, in order:

  1)  python src/demo_manual.py --watch
      Watch the arm pick up the red block. No AI involved yet.

  2)  python src/demo_manual.py --trials 10
      Run it 10 times. You want 10/10 before going further.

  3)  python src/run_episode.py --scenario B1 --provider fake
      Runs the full loop with a pretend AI. Still no API key needed.

  4)  Put an API key in your .env file, then:
      python src/run_episode.py --scenario B1 --provider anthropic
""")
