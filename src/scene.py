"""
scene.py — builds the MuJoCo world.

It reads scenarios.yaml, takes the objects a scenario asks for, and writes a
MuJoCo XML file describing the whole table: the robot arm, the table surface,
the blocks, the beaker, the bin, the tray, and the cylinder that stands in for
a person's hand.

You do not need to understand MuJoCo XML to use this. Call build_scene() and
you get back a loaded simulation.
"""

from pathlib import Path
import yaml
import mujoco

# --- where things live -------------------------------------------------------
# PROJECT_ROOT is the llm-robot-safety folder (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_FILE = PROJECT_ROOT / "scenarios.yaml"

# The robot model comes from Google DeepMind's "Menagerie" collection, which
# you download separately (see README step 3). We look for it next to the
# project folder first, then inside it.
MENAGERIE_CANDIDATES = [
    PROJECT_ROOT.parent / "mujoco_menagerie",
    PROJECT_ROOT / "mujoco_menagerie",
]

ARM_PATHS = {
    "panda": "franka_emika_panda/panda.xml",
    "ur5e": "universal_robots_ur5e/ur5e.xml",
}


def find_menagerie() -> Path:
    for c in MENAGERIE_CANDIDATES:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "Could not find the 'mujoco_menagerie' folder.\n"
        "Run this next to your project folder:\n"
        "  git clone https://github.com/google-deepmind/mujoco_menagerie.git\n"
        f"Looked in: {[str(c) for c in MENAGERIE_CANDIDATES]}"
    )


def load_config() -> dict:
    with open(SCENARIOS_FILE) as f:
        return yaml.safe_load(f)


# --- XML helpers -------------------------------------------------------------

def _free_box(name, pos, size, rgba, mass):
    hx, hy, hz = [s / 2 for s in size]
    return f"""
    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <freejoint name="{name}_free"/>
      <geom name="{name}_geom" type="box" size="{hx} {hy} {hz}"
            rgba="{rgba}" mass="{mass}" friction="1.2 0.02 0.001"
            solimp="0.99 0.999 0.001" solref="0.004 1"/>
    </body>"""


def _free_cylinder(name, pos, radius, height, rgba, mass):
    return f"""
    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <freejoint name="{name}_free"/>
      <geom name="{name}_geom" type="cylinder" size="{radius} {height/2}"
            rgba="{rgba}" mass="{mass}" friction="1.2 0.02 0.001"
            solimp="0.99 0.999 0.001" solref="0.004 1"/>
    </body>"""


def _fixed_cylinder(name, pos, radius, height, rgba):
    return f"""
    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <geom name="{name}_geom" type="cylinder" size="{radius} {height/2}"
            rgba="{rgba}" contype="1" conaffinity="1"/>
    </body>"""


def _open_container(name, pos, size, rgba):
    """A tray/bin: a floor plus four thin walls, so things can be dropped in."""
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    t = 0.006  # wall thickness
    return f"""
    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <geom name="{name}_floor" type="box" size="{hx} {hy} {t}" pos="0 0 {-hz}" rgba="{rgba}"/>
      <geom name="{name}_wx1" type="box" size="{t} {hy} {hz}" pos="{hx} 0 0" rgba="{rgba}"/>
      <geom name="{name}_wx2" type="box" size="{t} {hy} {hz}" pos="{-hx} 0 0" rgba="{rgba}"/>
      <geom name="{name}_wy1" type="box" size="{hx} {t} {hz}" pos="0 {hy} 0" rgba="{rgba}"/>
      <geom name="{name}_wy2" type="box" size="{hx} {t} {hz}" pos="0 {-hy} 0" rgba="{rgba}"/>
    </body>"""


COLOURS = {
    "red": "0.85 0.15 0.15 1",
    "blue": "0.15 0.30 0.85 1",
    "yellow": "0.90 0.80 0.10 1",
    "grey": "0.40 0.40 0.42 1",
    "white": "0.95 0.95 0.95 1",
    "transparent": "0.70 0.85 0.90 0.45",
}


def build_world_xml(object_ids, cfg, arm="panda", overrides=None) -> str:
    """Produce the full MuJoCo XML for one scenario's world."""
    overrides = overrides or {}
    menagerie = find_menagerie()
    arm_xml = menagerie / ARM_PATHS[arm]
    if not arm_xml.exists():
        raise FileNotFoundError(f"Robot model not found: {arm_xml}")

    bodies = []
    for oid in object_ids:
        spec = dict(cfg["objects"][oid])
        if "inherits" in spec:                      # labelled_beaker inherits beaker
            base = dict(cfg["objects"][spec["inherits"]])
            base.update({k: v for k, v in spec.items() if k != "inherits"})
            spec = base
        spec.update(overrides.get(oid, {}))

        pos = spec["pose"]
        rgba = COLOURS.get(spec.get("colour", "grey"), COLOURS["grey"])
        kind = spec["type"]

        if kind == "box":
            bodies.append(_free_box(oid, pos, spec["size"], rgba, spec.get("mass", 0.05)))
        elif kind == "cylinder":
            r, h = spec["size"]
            if spec.get("fixed"):
                bodies.append(_fixed_cylinder(oid, pos, r, h, "0.95 0.75 0.65 1"))
            else:
                bodies.append(_free_cylinder(oid, pos, r, h, rgba, spec.get("mass", 0.1)))
        elif kind == "container":
            bodies.append(_open_container(oid, pos, spec["size"], "0.35 0.35 0.38 1"))

    return f"""<mujoco model="llm_robot_safety_world">
  <include file="{arm_xml}"/>

  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global azimuth="140" elevation="-25" offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096" offsamples="8"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" width="512" height="512"
             rgb1="0.94 0.95 0.97" rgb2="0.78 0.82 0.88"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.80 0.81 0.84" rgb2="0.86 0.87 0.90"/>
    <material name="gridmat" texture="grid" texrepeat="6 6" reflectance="0.05"/>
    <material name="tablemat" rgba="0.82 0.78 0.72 1"/>
  </asset>

  <worldbody>
    <light pos="0.5 0 2.2" dir="0 0 -1" diffuse="0.75 0.75 0.75" ambient="0.45 0.45 0.47"/>
    <light pos="-0.4 -0.8 1.6" dir="0.3 0.5 -1" diffuse="0.25 0.25 0.28" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.05" pos="0 0 -0.75" material="gridmat"/>

    <!-- Table. Its TOP SURFACE is at z = 0, which is our reference height. -->
    <geom name="table" type="box" size="0.60 0.60 0.02" pos="0.45 0 -0.02"
          material="tablemat" friction="1.0 0.01 0.001"/>
    <geom name="table_leg" type="box" size="0.05 0.05 0.36" pos="0.45 0 -0.40"
          rgba="0.5 0.45 0.4 1" contype="0" conaffinity="0"/>
{"".join(bodies)}
  </worldbody>
</mujoco>
"""


def _prepare_build_dir(arm: str) -> Path:
    """MuJoCo looks for the robot's 3D mesh files in a folder called 'assets'
    next to the model file it loaded. We generate our world file into build/,
    so we link build/assets to the robot's real assets folder."""
    build = PROJECT_ROOT / "build"
    build.mkdir(exist_ok=True)
    src_assets = (find_menagerie() / ARM_PATHS[arm]).parent / "assets"
    dst_assets = build / "assets"
    if not dst_assets.exists():
        try:
            dst_assets.symlink_to(src_assets, target_is_directory=True)
        except OSError:                      # Windows without developer mode
            import shutil
            shutil.copytree(src_assets, dst_assets)
    return build


def build_scene(object_ids, arm="panda", overrides=None, cfg=None,
                jitter=0.0, seed=None):
    """Build the world and return (model, data, cfg).

    model  = the description of the world
    data   = the current state of the world (positions, velocities, ...)
    jitter = randomly nudge each object by up to this many metres. Use a small
             value (e.g. 0.015) when repeating a trial, otherwise every run is
             identical and repeating it tells you nothing.
    """
    cfg = cfg or load_config()

    if jitter > 0:
        import random
        rng = random.Random(seed)
        overrides = {k: dict(v) for k, v in (overrides or {}).items()}
        for oid in object_ids:
            spec = cfg["objects"][oid]
            if spec.get("fixed") or spec.get("type") == "container":
                continue                    # bins, trays and the human stay put
            base = overrides.get(oid, {}).get("pose") or spec.get(
                "pose", cfg["objects"].get(spec.get("inherits", oid), {}).get("pose"))
            if base is None:
                continue
            overrides.setdefault(oid, {})["pose"] = [
                base[0] + rng.uniform(-jitter, jitter),
                base[1] + rng.uniform(-jitter, jitter),
                base[2],
            ]

    xml = build_world_xml(object_ids, cfg, arm=arm, overrides=overrides)
    build = _prepare_build_dir(arm)
    world_file = build / "world.xml"
    world_file.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(world_file))
    data = mujoco.MjData(model)

    # Put the ARM in its "home" pose, without disturbing the objects.
    # (We cannot use mj_resetDataKeyframe here: the robot's keyframe only
    # describes the 9 arm/finger joints, so it would teleport every object on
    # the table to the origin.)
    mujoco.mj_resetData(model, data)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        data.qpos[:9] = model.key_qpos[key][:9]
        data.ctrl[:] = model.key_ctrl[key]
    mujoco.mj_forward(model, data)
    return model, data, cfg


def save_world_xml(object_ids, out_path, arm="panda", overrides=None):
    """Write the generated XML to a file, so you can open it and look at it."""
    cfg = load_config()
    Path(out_path).write_text(build_world_xml(object_ids, cfg, arm, overrides))
    return out_path


if __name__ == "__main__":
    m, d, _ = build_scene(["red_block", "blue_block", "beaker", "tray", "bin", "human_proxy"])
    print(f"World built OK.  bodies={m.nbody}  joints={m.njnt}  actuators={m.nu}")
