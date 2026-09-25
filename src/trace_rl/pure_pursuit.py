"""Hand-written baseline: pure pursuit on the centreline plus a curvature-based speed profile.

It uses privileged information (car pose, full track) that the policy never sees. Its job is to
prove the env is drivable (M3) and to be the lap-time baseline the policy must beat (M5).
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from trace_rl.env import EnvState, interp
from trace_rl.physics import RHO_AIR, CarParams, G
from trace_rl.track import MAX_POINTS, Track


@dataclass(frozen=True)
class PPConfig:
    grip_frac: float = 0.8  # fraction of peak lateral grip the speed profile uses
    brake_frac: float = 0.6  # fraction of peak grip used for braking in the speed profile
    v_max: float = 85.0  # m/s
    lookahead_gain: float = 0.35  # s; steering target distance = gain * speed
    lookahead_min: float = 6.0  # m
    lookahead_max: float = 40.0  # m
    speed_preview: float = 0.3  # s; speed target is read this far ahead to cover actuation lag
    speed_gain: float = 0.5  # pedal per m/s of speed error
    traction_frac: float = 0.8  # share of the rear grip left after cornering that throttle may use


def speed_profile(track: dict, params: CarParams, cfg: PPConfig = PPConfig()) -> np.ndarray:
    """Max corner speed from curvature, then a backward pass so braking starts early enough."""
    k = np.maximum(np.abs(track["curvature"]), 1e-4)
    # Grip per unit mass is mu (g + ka v^2) with downforce, so k v^2 = f mu (g + ka v^2).
    ka = 0.5 * RHO_AIR * params.cla / params.mass
    fmu = cfg.grip_frac * params.mu
    denom = k - fmu * ka
    v = np.where(denom > 0, np.sqrt(fmu * G / np.maximum(denom, 1e-9)), cfg.v_max)
    v = np.minimum(v, cfg.v_max)
    n = track["n"]
    for _ in range(2):  # two passes so braking zones wrap across the start line
        for i in range(n - 1, -1, -1):
            # Downforce at the slower exit speed, so the braking estimate stays conservative.
            v_next = v[(i + 1) % n]
            a_brake = cfg.brake_frac * params.mu * (G + ka * v_next**2)
            v[i] = min(v[i], np.sqrt(v_next**2 + 2 * a_brake * track["ds"]))
    out = np.zeros(MAX_POINTS, np.float32)
    out[:n] = v
    return out


def act(
    state: EnvState,
    tracks: Track,
    v_profile: jnp.ndarray,
    params: CarParams,
    cfg: PPConfig = PPConfig(),
):
    """v_profile: (K, MAX_POINTS) speed profiles matching `tracks`."""
    car, tid = state.car, state.track_id
    ld = jnp.clip(cfg.lookahead_gain * car.vx, cfg.lookahead_min, cfg.lookahead_max)
    s_target = state.s + ld
    tx = interp(tracks.xy[..., 0], tracks, tid, s_target)
    ty = interp(tracks.xy[..., 1], tracks, tid, s_target)

    # Pure pursuit geometry is about the rear axle.
    c, sn = jnp.cos(car.yaw), jnp.sin(car.yaw)
    dx, dy = tx - (car.x - params.lr * c), ty - (car.y - params.lr * sn)
    local_x, local_y = c * dx + sn * dy, -sn * dx + c * dy
    dist2 = local_x**2 + local_y**2
    delta = jnp.arctan(2 * (params.lf + params.lr) * local_y / dist2)
    steer = jnp.clip(delta / params.max_steer, -1.0, 1.0)

    v_target = interp(v_profile, tracks, tid, state.s + cfg.speed_preview * car.vx)
    err = v_target - car.vx
    # Traction budget: rear grip left after the lateral load of the current corner. Without it the
    # car spins on corner exit (370 kW, rear drive).
    wheelbase = params.lf + params.lr
    fz = params.mass * G + 0.5 * RHO_AIR * params.cla * car.vx**2
    rear_grip = params.mu * fz * params.lf / wheelbase
    rear_lat = params.mass * jnp.abs(car.vx * car.r) * params.lf / wheelbase
    budget = cfg.traction_frac * jnp.sqrt(jnp.maximum(rear_grip**2 - rear_lat**2, 0.0))
    max_drive = jnp.minimum(params.max_drive_force, params.power / jnp.maximum(car.vx, 1.0))
    throttle = jnp.clip(cfg.speed_gain * err, 0.0, budget / max_drive)
    brake = jnp.clip(-cfg.speed_gain * err, 0.0, 1.0)
    return jnp.stack([steer, throttle, brake])
