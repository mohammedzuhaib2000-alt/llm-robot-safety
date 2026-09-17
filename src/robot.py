"""
robot.py — the robot's "menu" of allowed commands.

This is the layer between the LLM and the simulator. The LLM is only ever
allowed to call the functions in here. Anything not in this file, the robot
cannot do — which is itself a safety property, and worth saying so in the
dissertation.

Nothing in this file knows anything about LLMs. You can (and should) test it
on its own first, with demo_manual.py.
"""

import numpy as np
import mujoco

# The Panda's "tool centre point" (the spot between the fingertips) sits about
# 10.3 cm out from the hand body, along the hand's own z axis.
TCP_OFFSET = np.array([0.0, 0.0, 0.1034])

# Gripper actuator range: 255 = fully open, 0 = fully closed.
GRIP_OPEN = 255.0
GRIP_CLOSED = 0.0


class ActionResult:
    """What every command hands back: did it work, and why not if it didn't."""

    def __init__(self, ok, message, **extra):
        self.ok = ok
        self.message = message
        self.extra = extra

    def __repr__(self):
        return f"{'OK ' if self.ok else 'FAIL'}: {self.message}"

    def to_dict(self):
        return {"ok": self.ok, "message": self.message, **self.extra}


class Robot:
    def __init__(self, model, data, cfg, viewer=None):
        self.m = model
        self.d = data
        self.cfg = cfg
        self.viewer = viewer

        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self.hand_id < 0:
            raise RuntimeError("No body named 'hand' — is this the Panda model?")

        # The 7 arm joints are actuators 0..6; actuator 7 is the gripper.
        self.arm_actuators = list(range(7))
        self.grip_actuator = 7
        self.arm_dofs = list(range(7))

        self.held_object = None
        self.max_speed_seen = 0.0
        self.log = []            # every command, in order — your evidence trail

    # ---------------------------------------------------------------- sensing

    def tcp_pos(self):
        """Where the gripper's grasp point currently is, in metres."""
        R = self.d.xmat[self.hand_id].reshape(3, 3)
        return self.d.xpos[self.hand_id] + R @ TCP_OFFSET

    def tcp_speed(self):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.hand_id, vel, 0
        )
        return float(np.linalg.norm(vel[3:6]))

    def object_pos(self, name):
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return None
        return np.array(self.d.xpos[bid])

    def gripper_opening(self):
        """Distance between the fingers, in metres (0.08 = fully open)."""
        return float(self.d.qpos[-2] + self.d.qpos[-1])

    def scene_objects(self):
        """List every object on the table, with its position."""
        out = []
        for i in range(self.m.nbody):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
            if name in (None, "world") or name.startswith("link") or name in (
                "hand", "left_finger", "right_finger"
            ):
                continue
            out.append((name, np.array(self.d.xpos[i])))
        return out

    # -------------------------------------------------------- inverse kinemat.

    def _ik(self, target_pos, target_z_axis=(0, 0, -1), iters=120):
        """Work out joint angles that put the gripper at target_pos, pointing
        the given way. Uses damped least squares — a standard, forgiving method.

        Returns joint angles, or None if it could not get close enough.
        """
        q = self.d.qpos.copy()
        d_ik = mujoco.MjData(self.m)
        d_ik.qpos[:] = q
        target_pos = np.asarray(target_pos, dtype=float)
        target_z = np.asarray(target_z_axis, dtype=float)
        target_z = target_z / np.linalg.norm(target_z)

        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))

        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d_ik)
            mujoco.mj_comPos(self.m, d_ik)

            R = d_ik.xmat[self.hand_id].reshape(3, 3)
            tcp = d_ik.xpos[self.hand_id] + R @ TCP_OFFSET
            pos_err = target_pos - tcp

            # Orientation: rotate the hand's z axis onto the target axis.
            cur_z = R @ np.array([0.0, 0.0, 1.0])
            rot_err = np.cross(cur_z, target_z)

            err = np.concatenate([pos_err, rot_err * 0.6])
            if np.linalg.norm(pos_err) < 1e-3 and np.linalg.norm(rot_err) < 0.03:
                break

            mujoco.mj_jac(self.m, d_ik, jacp, jacr, tcp, self.hand_id)
            J = np.vstack([jacp[:, self.arm_dofs], jacr[:, self.arm_dofs]])

            lam = 0.12                                   # damping: stability > speed
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
            dq = np.clip(dq, -0.25, 0.25)
            d_ik.qpos[self.arm_dofs] += dq

            # respect the arm's joint limits
            for j in self.arm_dofs:
                lo, hi = self.m.jnt_range[j]
                d_ik.qpos[j] = np.clip(d_ik.qpos[j], lo, hi)

        mujoco.mj_kinematics(self.m, d_ik)
        R = d_ik.xmat[self.hand_id].reshape(3, 3)
        tcp = d_ik.xpos[self.hand_id] + R @ TCP_OFFSET
        if np.linalg.norm(target_pos - tcp) > 0.02:
            return None
        return d_ik.qpos[self.arm_dofs].copy()

    # ------------------------------------------------------------- simulation

    def _step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            self.max_speed_seen = max(self.max_speed_seen, self.tcp_speed())
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

    def _settle(self, steps=200):
        self._step(steps)

    # ---------------------------------------------------------------- actions

    def move_to(self, x, y, z, speed=0.25, z_axis=(0, 0, -1)):
        """Move the gripper to a position, travelling in a straight line at
        roughly `speed` metres per second."""
        target = np.array([x, y, z], dtype=float)
        start = self.tcp_pos()
        dist = float(np.linalg.norm(target - start))

        if self._ik(target, z_axis) is None:
            r = ActionResult(False, f"cannot reach ({x:.3f}, {y:.3f}, {z:.3f}) — out of range")
            self.log.append(("move_to", dict(x=x, y=y, z=z, speed=speed), r.to_dict()))
            return r

        # Break the straight line into small steps so speed is controllable.
        n_way = max(2, int(dist / max(speed, 0.01) / 0.05))
        for i in range(1, n_way + 1):
            wp = start + (target - start) * (i / n_way)
            q = self._ik(wp, z_axis)
            if q is not None:
                self.d.ctrl[self.arm_actuators] = q
            self._step(40)

        self._step(60)
        err = float(np.linalg.norm(self.tcp_pos() - target))
        r = ActionResult(err < 0.03, f"moved to ({x:.3f}, {y:.3f}, {z:.3f}), error {err*1000:.0f}mm",
                         position_error_m=err)
        self.log.append(("move_to", dict(x=x, y=y, z=z, speed=speed), r.to_dict()))
        return r

    def open_gripper(self):
        self.d.ctrl[self.grip_actuator] = GRIP_OPEN
        self._step(300)
        self.held_object = None
        r = ActionResult(True, "gripper open")
        self.log.append(("open_gripper", {}, r.to_dict()))
        return r

    def close_gripper(self, force=60.0):
        """Close the fingers. `force` caps how hard the gripper may squeeze —
        this is what scenario B10 ('pick it up gently') measures."""
        force = float(np.clip(force, 1.0, 100.0))
        self.m.actuator_forcerange[self.grip_actuator] = [-force, force]
        self.d.ctrl[self.grip_actuator] = GRIP_CLOSED
        self._step(400)
        r = ActionResult(True, f"gripper closed at {force:.0f}N, opening {self.gripper_opening()*1000:.0f}mm",
                         force_N=force, opening_m=self.gripper_opening())
        self.log.append(("close_gripper", dict(force=force), r.to_dict()))
        return r

    def pick(self, object_id, force=60.0):
        """Pick an object up: approach from above, close, lift."""
        pos = self.object_pos(object_id)
        if pos is None:
            r = ActionResult(False, f"there is no object called '{object_id}'")
            self.log.append(("pick", dict(object_id=object_id), r.to_dict()))
            return r
        if self.held_object is not None:
            r = ActionResult(False, f"already holding '{self.held_object}'")
            self.log.append(("pick", dict(object_id=object_id), r.to_dict()))
            return r

        x, y, z = pos
        self.open_gripper()
        approach = self.move_to(x, y, z + 0.12, speed=0.30)
        if not approach.ok:
            self.log.append(("pick", dict(object_id=object_id), approach.to_dict()))
            return approach
        self.move_to(x, y, z + 0.005, speed=0.10)
        self.close_gripper(force)
        self.move_to(x, y, z + 0.20, speed=0.15)

        lifted = bool(self.object_pos(object_id)[2] > z + 0.05)
        self.held_object = object_id if lifted else None
        r = ActionResult(lifted,
                         f"picked up '{object_id}'" if lifted
                         else f"tried to pick '{object_id}' but the grasp failed")
        self.log.append(("pick", dict(object_id=object_id, force=force), r.to_dict()))
        return r

    def place(self, object_id, target):
        """Put the held object down. `target` is another object's name
        (e.g. 'tray') or an (x, y, z) position."""
        if self.held_object != object_id:
            r = ActionResult(False, f"not holding '{object_id}'")
            self.log.append(("place", dict(object_id=object_id, target=str(target)), r.to_dict()))
            return r

        if isinstance(target, str):
            tpos = self.object_pos(target)
            if tpos is None:
                r = ActionResult(False, f"there is no place called '{target}'")
                self.log.append(("place", dict(object_id=object_id, target=target), r.to_dict()))
                return r
            drop = [tpos[0], tpos[1], tpos[2] + 0.16]
        else:
            drop = list(target)

        self.move_to(drop[0], drop[1], max(drop[2], 0.14), speed=0.25)
        self.move_to(drop[0], drop[1], max(drop[2] - 0.04, 0.10), speed=0.10)
        self.open_gripper()
        self.move_to(drop[0], drop[1], drop[2] + 0.12, speed=0.25)
        self._settle(400)

        self.held_object = None
        r = ActionResult(True, f"placed '{object_id}' at {target}")
        self.log.append(("place", dict(object_id=object_id, target=str(target)), r.to_dict()))
        return r

    def report(self, message):
        r = ActionResult(True, f"reported: {message}")
        self.log.append(("report", dict(message=message), r.to_dict()))
        return r

    def home(self):
        key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            self.d.ctrl[self.arm_actuators] = self.m.key_qpos[key][:7]
        self._step(500)
        return ActionResult(True, "returned to home position")
