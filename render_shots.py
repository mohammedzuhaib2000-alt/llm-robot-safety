"""Render still images of the simulation for the report."""
import os
import sys
import pathlib
import numpy as np
import mujoco
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
from scene import build_scene, load_config          # noqa: E402
from robot import Robot                             # noqa: E402

OUT = pathlib.Path("/home/claude/shots")
OUT.mkdir(exist_ok=True)
W, H = 1400, 900


def shot(model, data, name, azimuth=138, elevation=-24, distance=1.35,
         lookat=(0.45, 0.0, 0.10)):
    r = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    cam.lookat[:] = lookat
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    r.update_scene(data, cam, opt)
    img = r.render()
    Image.fromarray(img).save(OUT / f"{name}.png")
    r.close()
    print("saved", name)


cfg = load_config()

# --- 1. the whole workspace -------------------------------------------------
objs = ["red_block", "blue_block", "yellow_block", "heavy_block",
        "beaker", "bin", "tray", "human_proxy"]
m, d, cfg = build_scene(objs, cfg=cfg)
bot = Robot(m, d, cfg)
bot._settle(400)
shot(m, d, "01_workspace")

# --- 2. mid-grasp: holding the red block ------------------------------------
bot.pick("red_block")
shot(m, d, "02_holding_block", azimuth=150, elevation=-20, distance=1.15,
     lookat=(0.42, 0.05, 0.18))

# --- 3. placed in the tray --------------------------------------------------
bot.place("red_block", "tray")
shot(m, d, "03_placed", azimuth=150, elevation=-28, distance=1.25,
     lookat=(0.40, 0.15, 0.08))

# --- 4. the note-card attack scene (A6) -------------------------------------
m2, d2, cfg = build_scene(["red_block", "blue_block", "bin", "tray", "note_card"],
                          cfg=cfg,
                          overrides={"note_card": {"size": [0.15, 0.11, 0.004],
                                                   "pose": [0.44, 0.02, 0.003],
                                                   "colour": "white"}})
b2 = Robot(m2, d2, cfg)
b2._settle(400)
shot(m2, d2, "04_notecard", azimuth=148, elevation=-38, distance=0.80,
     lookat=(0.44, 0.02, 0.03))

# --- 5. close view of the operator zone (B9-style) --------------------------
m3, d3, cfg = build_scene(["red_block", "human_proxy", "tray"], cfg=cfg,
                          overrides={"red_block": {"pose": [0.56, 0.26, 0.02]}})
b3 = Robot(m3, d3, cfg)
b3._settle(400)
shot(m3, d3, "05_operator", azimuth=160, elevation=-22, distance=1.0,
     lookat=(0.52, 0.25, 0.10))

print("done")
