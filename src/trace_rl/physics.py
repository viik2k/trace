"""M1: dynamic bicycle model with simplified Pacejka tyres.

SI units. Body frame: x forward, y left, yaw counter-clockwise positive.
Action: [steer in [-1, 1], throttle in [0, 1], brake in [0, 1]].
"""

import equinox as eqx
import jax
import jax.numpy as jnp

G = 9.81
RHO_AIR = 1.225
# Below V_KIN total speed the car follows the kinematic bicycle; above V_DYN the dynamic one;
# linear blend between. Slip angles are ill-conditioned near standstill and make the dynamic
# model stiff there. Total speed, not vx: a car sliding sideways must keep its momentum.
V_KIN = 2.0
V_DYN = 5.0


class CarParams(eqx.Module):
    """Generic GT3-ish car. An eqx.Module is a pytree: every field is a leaf, so a batch of cars
    is one CarParams whose fields have a leading axis, and jax.vmap maps over it directly."""

    mass: float = 1300.0  # kg
    iz: float = 2200.0  # yaw inertia, kg m^2
    lf: float = 1.5  # CG to front axle, m (44% static front weight)
    lr: float = 1.2  # CG to rear axle, m
    h_cg: float = 0.45  # m, only used when load_transfer > 0
    max_steer: float = 0.35  # road-wheel angle at steer = 1, rad
    power: float = 370e3  # W at the wheels, rear-wheel drive
    max_drive_force: float = 9000.0  # N, caps drive force at low speed
    max_brake_force: float = 20000.0  # N, both axles
    brake_bias_front: float = 0.6
    cda: float = 1.2  # drag area, m^2
    crr: float = 0.015  # rolling resistance coefficient
    mu: float = 1.5  # peak tyre friction
    # Pacejka lateral: Fy = mu Fz sin(C atan(B a - E (B a - atan(B a)))).
    # Rear slightly stiffer than front gives mild understeer instead of knife-edge neutral steer.
    pac_b_front: float = 11.0
    pac_b_rear: float = 13.0
    pac_c: float = 1.9
    pac_e: float = 0.97
    # 0 = off, 1 = on. A float rather than a bool so the pytree stays all-numeric for vmap.
    load_transfer: float = 0.0
    dt: float = 1 / 60
    # Static: part of the pytree structure, not a leaf. It sets a Python loop count, which must be
    # known at trace time.
    n_substeps: int = eqx.field(static=True, default=4)


class CarState(eqx.Module):
    x: jax.Array  # world position, m
    y: jax.Array
    yaw: jax.Array  # rad, wrapped to [-pi, pi)
    vx: jax.Array  # body-frame velocity, m/s (vx >= 0, no reverse gear)
    vy: jax.Array
    r: jax.Array  # yaw rate, rad/s


def init_state(x=0.0, y=0.0, yaw=0.0, vx=0.0) -> CarState:
    f = lambda v: jnp.asarray(v, jnp.float32)  # noqa: E731
    return CarState(x=f(x), y=f(y), yaw=f(yaw), vx=f(vx), vy=f(0.0), r=f(0.0))


def slip_angles(s: CarState, steer_angle, p: CarParams):
    # vx floored so slip stays bounded at standstill; the kinematic blend takes over there anyway.
    vx = jnp.maximum(s.vx, 1.0)
    alpha_f = steer_angle - jnp.arctan2(s.vy + p.lf * s.r, vx)
    alpha_r = -jnp.arctan2(s.vy - p.lr * s.r, vx)
    return alpha_f, alpha_r


def _pacejka(alpha, fz, b, p: CarParams):
    ba = b * alpha
    return p.mu * fz * jnp.sin(p.pac_c * jnp.arctan(ba - p.pac_e * (ba - jnp.arctan(ba))))


def _friction_circle(fx, fy, fz, mu):
    # Scale both components so the combined force stays within mu * Fz. Without this the car can
    # brake and corner at full grip simultaneously.
    f = jnp.sqrt(fx**2 + fy**2 + 1e-6)
    k = jnp.minimum(1.0, mu * fz / f)
    return fx * k, fy * k


def _accelerations(s: CarState, delta, throttle, brake, p: CarParams):
    """Returns dynamic-model (ax, ay, yaw_acc) and the kinematic-model ax."""
    wheelbase = p.lf + p.lr
    stop = jnp.tanh(s.vx / 0.5)  # smooth sign(vx) so brakes and rolling resistance never reverse
    drive = throttle * jnp.minimum(p.max_drive_force, p.power / jnp.maximum(s.vx, 1.0))
    brake_force = brake * p.max_brake_force * stop
    fx_f = -p.brake_bias_front * brake_force
    fx_r = drive - (1.0 - p.brake_bias_front) * brake_force
    resist = 0.5 * RHO_AIR * p.cda * s.vx**2 + p.crr * p.mass * G * stop

    # Longitudinal load transfer from the commanded (pre-saturation) acceleration.
    dfz = p.load_transfer * (fx_f + fx_r - resist) * p.h_cg / wheelbase
    fz_f = jnp.maximum(p.mass * G * p.lr / wheelbase - dfz, 0.0)
    fz_r = jnp.maximum(p.mass * G * p.lf / wheelbase + dfz, 0.0)

    # Kinematic regime: no lateral tyre forces, longitudinal force limited by grip alone.
    ax_kin = (
        jnp.clip(fx_f, -p.mu * fz_f, p.mu * fz_f)
        + jnp.clip(fx_r, -p.mu * fz_r, p.mu * fz_r)
        - resist
    ) / p.mass

    alpha_f, alpha_r = slip_angles(s, delta, p)
    fx_f, fy_f = _friction_circle(fx_f, _pacejka(alpha_f, fz_f, p.pac_b_front, p), fz_f, p.mu)
    fx_r, fy_r = _friction_circle(fx_r, _pacejka(alpha_r, fz_r, p.pac_b_rear, p), fz_r, p.mu)

    cd, sd = jnp.cos(delta), jnp.sin(delta)
    ax = (fx_r + fx_f * cd - fy_f * sd - resist) / p.mass + s.vy * s.r
    ay = (fy_r + fx_f * sd + fy_f * cd) / p.mass - s.vx * s.r
    yaw_acc = (p.lf * (fy_f * cd + fx_f * sd) - p.lr * fy_r) / p.iz
    return ax, ay, yaw_acc, ax_kin


def _substep(s: CarState, delta, throttle, brake, p: CarParams, h) -> CarState:
    # Semi-implicit Euler: velocities first, then positions from the new velocities.
    ax, ay, yaw_acc, ax_kin = _accelerations(s, delta, throttle, brake, p)
    tan_d = jnp.tan(delta)
    wheelbase = p.lf + p.lr
    w = jnp.clip((jnp.hypot(s.vx, s.vy) - V_KIN) / (V_DYN - V_KIN), 0.0, 1.0)

    vx = jnp.maximum(s.vx + h * (w * ax + (1.0 - w) * ax_kin), 0.0)
    # Kinematic bicycle: vy and yaw rate are fixed by speed and steer (no tyre slip).
    vy = w * (s.vy + h * ay) + (1.0 - w) * vx * p.lr * tan_d / wheelbase
    r = w * (s.r + h * yaw_acc) + (1.0 - w) * vx * tan_d / wheelbase

    yaw = s.yaw + h * r
    c, sn = jnp.cos(yaw), jnp.sin(yaw)
    x = s.x + h * (vx * c - vy * sn)
    y = s.y + h * (vx * sn + vy * c)
    yaw = (yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return CarState(x=x, y=y, yaw=yaw, vx=vx, vy=vy, r=r)


def step(state: CarState, action: jax.Array, params: CarParams) -> CarState:
    """Advance one dt. Pure: no side effects, safe to jit and vmap over state, action and params."""
    delta = jnp.clip(action[0], -1.0, 1.0) * params.max_steer
    throttle = jnp.clip(action[1], 0.0, 1.0)
    brake = jnp.clip(action[2], 0.0, 1.0)
    h = params.dt / params.n_substeps
    # Plain Python loop: n_substeps is static, so this unrolls at trace time. The barrier stops XLA
    # fusing across substeps: without it the fusion pass duplicates shared subexpressions down the
    # unrolled chain and cost grows super-linearly (measured 500x slower at 12 substeps/decision).
    for _ in range(params.n_substeps):
        state = jax.lax.optimization_barrier(_substep(state, delta, throttle, brake, params, h))
    return state
