"""Rerun logging for tracks and driving trajectories.

Writes .rrd files; open with `uv run rerun <file>.rrd` (on WSL2 the Windows Rerun viewer can open
the file too). Logged in 3D at z = 0 because Rerun's 2D views put +y down, mirroring the track.
"""

import numpy as np
import rerun as rr


def start(path, app_id: str = "trace") -> None:
    rr.init(app_id)
    rr.save(path)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def _z(xy):
    return np.column_stack([xy, np.zeros(len(xy))])


def log_track(prefix: str, tr: dict) -> None:
    """tr: unpadded numpy track dict (see track.to_numpy)."""
    xy, h, w = tr["xy"], tr["heading"], tr["width"]
    normal = np.stack([-np.sin(h), np.cos(h)], axis=1)
    closed = lambda a: _z(np.vstack([a, a[:1]]))  # noqa: E731
    left, right = xy + normal * w[:, None] / 2, xy - normal * w[:, None] / 2
    rr.log(
        f"{prefix}/edges",
        rr.LineStrips3D([closed(left), closed(right)], colors=[220, 220, 220], radii=0.3),
        static=True,
    )
    rr.log(
        f"{prefix}/centreline",
        rr.LineStrips3D([closed(xy)], colors=[110, 110, 110], radii=0.08),
        static=True,
    )
    rr.log(f"{prefix}/start", rr.Points3D(_z(xy[:1]), colors=[255, 60, 60], radii=2.0), static=True)


def _speed_colors(v, v_max=80.0):
    t = np.clip(v / v_max, 0, 1)[:, None]
    return (np.array([40, 90, 255]) * (1 - t) + np.array([255, 60, 30]) * t).astype(np.uint8)


def log_trajectory(prefix: str, traj: dict, decision_dt: float, color=(255, 200, 0)) -> None:
    """traj: numpy rollout dict from env.rollout. Logs the path coloured by speed (static) and an
    animated car on the `time` timeline."""
    alive = np.asarray(traj["alive"])
    n = int(alive.sum()) if not alive.all() else len(alive)
    xy = np.stack([traj["x"][:n], traj["y"][:n]], axis=1)
    speed, yaw, over = traj["speed"][:n], traj["yaw"][:n], traj["over_limits"][:n]
    rr.log(
        f"{prefix}/path", rr.Points3D(_z(xy), colors=_speed_colors(speed), radii=0.4), static=True
    )
    if over.any():
        rr.log(
            f"{prefix}/offtrack",
            rr.Points3D(_z(xy[over]), colors=[255, 0, 255], radii=1.0),
            static=True,
        )
    for i in range(n):
        rr.set_time("time", duration=(i + 1) * decision_dt)
        rr.log(f"{prefix}/car", rr.Points3D(_z(xy[i : i + 1]), colors=[color], radii=1.5))
        heading = 5.0 * np.array([[np.cos(yaw[i]), np.sin(yaw[i]), 0.0]])
        rr.log(
            f"{prefix}/car_heading",
            rr.Arrows3D(origins=_z(xy[i : i + 1]), vectors=heading, colors=[color]),
        )
        rr.log(f"{prefix}/speed_kmh", rr.Scalars(speed[i] * 3.6))
        rr.log(f"{prefix}/action", rr.Scalars(np.asarray(traj["action"][i])))
