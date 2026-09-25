"""M3: racing environment as pure functions over pytrees.

Nothing here is jitted directly; callers jit/vmap at the top (the PPO training loop, tests).
`tracks` is always a batched Track (leading axis K) and envs index it by track_id.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from trace_rl import physics
from trace_rl.physics import CarParams, CarState
from trace_rl.track import Track


@dataclass(frozen=True)
class EnvConfig:
    # Frozen dataclass: hashable, so it can be a static argument to jit. Its values set array
    # shapes (n_lookahead) and Python loop counts (action_repeat), fixed at trace time.
    n_lookahead: int = 20
    lookahead_spacing: float = 10.0  # m, so 20 samples see 200 m ahead
    action_repeat: int = 3  # physics steps per decision: 60 Hz physics, 20 Hz policy
    max_steps: int = 3000  # decisions per episode (150 s)
    # Reward: progress per metre, minus penalties. Scaled so returns stay O(1-10) for the critic.
    progress_scale: float = 0.01  # per metre of progress
    # Per decision past track limits (= 20 m of progress). Was 0.05: at that price 1.5B-step
    # policies cut the hairpin 4-16 times per 4 laps (2026-09-26).
    offtrack_penalty: float = 0.2
    jerk_penalty: float = 0.01  # times ||a_t - a_{t-1}||^2 (full steer flip = 4 m)
    # On any termination, stuck included, so stopping is never a way to dodge it. Off: the GPU A/B
    # (2026-09-25) showed no clean-lap gain from 2.0 and ~12% slower driving.
    crash_penalty: float = 0.0
    car_half_width: float = 1.0  # m
    runoff: float = 3.0  # m beyond track limits before termination
    stuck_speed: float = 1.0  # m/s
    stuck_time: float = 2.0  # s
    start_speed: tuple[float, float] = (0.0, 20.0)  # m/s, random resets; includes standing starts
    search_window: int = 8  # centreline points either side of the last index

    @property
    def obs_dim(self) -> int:
        return 10 + 2 * self.n_lookahead


class EnvState(eqx.Module):
    car: CarState
    track_id: jax.Array  # () int32
    idx: jax.Array  # () int32 nearest centreline point; seeds the next local search
    s: jax.Array  # () arc length of the car's projection onto the centreline, [0, length)
    progress: jax.Array  # () metres along the track since reset; negative if driving backwards
    prev_action: jax.Array  # (3,)
    t: jax.Array  # () decisions since reset
    stuck: jax.Array  # () consecutive decisions below stuck_speed
    offtrack: jax.Array  # () decisions past track limits this episode
    ep_return: jax.Array  # () reward summed this episode


ACTION_LOW = jnp.array([-1.0, 0.0, 0.0])
ACTION_HIGH = jnp.array([1.0, 1.0, 1.0])


# --- Track queries ------------------------------------------------------------------------------


def interp(field: jax.Array, tracks: Track, tid, s):
    """Linear interpolation of a (K, M) track field at arc length s (scalar or array), wrapping."""
    u = s / tracks.ds[tid]
    i0 = jnp.floor(u).astype(jnp.int32)
    f = u - i0
    n = tracks.n[tid]
    return field[tid, i0 % n] * (1 - f) + field[tid, (i0 + 1) % n] * f


def frame(tracks: Track, tid, i, car: CarState):
    """Car pose relative to centreline point i: (along-track offset, lateral offset, heading error).
    Lateral is positive to the left of the centreline."""
    h = tracks.heading[tid, i]
    rel = jnp.stack([car.x, car.y]) - tracks.xy[tid, i]
    along = rel[0] * jnp.cos(h) + rel[1] * jnp.sin(h)
    lateral = -rel[0] * jnp.sin(h) + rel[1] * jnp.cos(h)
    heading_err = (car.yaw - h + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return along, lateral, heading_err


def nearest(tracks: Track, tid, idx, car: CarState, window: int):
    """Nearest centreline point, searching only near the previous one. A global search could jump
    to a different section of track that passes close by; a local one can't."""
    n = tracks.n[tid]
    cand = (idx + jnp.arange(-window, window + 1)) % n
    d2 = ((tracks.xy[tid, cand] - jnp.stack([car.x, car.y])) ** 2).sum(-1)
    return cand[jnp.argmin(d2)]


# --- Env API ------------------------------------------------------------------------------------


def observe(state: EnvState, tracks: Track, params: CarParams, cfg: EnvConfig) -> jax.Array:
    """Car-frame observation. No global position or heading, so nothing ties the policy to a
    particular track's coordinates."""
    car, tid = state.car, state.track_id
    _, lateral, heading_err = frame(tracks, tid, state.idx, car)
    half_w = 0.5 * interp(tracks.width, tracks, tid, state.s)
    alpha_f, alpha_r = physics.slip_angles(car, state.prev_action[0] * params.max_steer, params)
    s_ahead = state.s + cfg.lookahead_spacing * jnp.arange(cfg.n_lookahead)
    # Fixed hand scaling to roughly unit range. No running statistics to carry around.
    return jnp.concatenate(
        [
            jnp.stack(
                [
                    car.vx / 30.0,
                    car.vy / 5.0,
                    car.r,
                    alpha_f / 0.2,
                    alpha_r / 0.2,
                    lateral / half_w,
                    heading_err,
                ]
            ),
            state.prev_action,
            interp(tracks.curvature, tracks, tid, s_ahead) * 20.0,  # 15 m hairpin -> 1.3
            (interp(tracks.width, tracks, tid, s_ahead) - 12.0) / 4.0,
        ]
    )


def init_at(
    tracks: Track,
    tid,
    idx,
    params: CarParams,
    cfg: EnvConfig,
    lateral=0.0,
    heading_offset=0.0,
    speed=0.0,
):
    """Deterministic start: car at centreline point idx, offset sideways, pointing along track."""
    h = tracks.heading[tid, idx]
    p = tracks.xy[tid, idx]
    car = physics.init_state(
        p[0] - jnp.sin(h) * lateral, p[1] + jnp.cos(h) * lateral, h + heading_offset, speed
    )
    zero_i, zero_f = jnp.int32(0), jnp.float32(0.0)
    state = EnvState(
        car=car,
        track_id=jnp.int32(tid),
        idx=jnp.int32(idx),
        s=tracks.s[tid, idx],
        progress=zero_f,
        prev_action=jnp.zeros(3),
        t=zero_i,
        stuck=zero_i,
        offtrack=zero_i,
        ep_return=zero_f,
    )
    return state, observe(state, tracks, params, cfg)


def reset(key, tracks: Track, params: CarParams, cfg: EnvConfig):
    """Random track, random point on it, small lateral and heading noise, rolling start."""
    k_track, k_idx, k_lat, k_head, k_speed = jax.random.split(key, 5)
    tid = jax.random.randint(k_track, (), 0, tracks.n.shape[0])
    idx = jax.random.randint(k_idx, (), 0, tracks.n[tid])
    half_w = 0.5 * tracks.width[tid, idx]
    return init_at(
        tracks,
        tid,
        idx,
        params,
        cfg,
        lateral=jax.random.uniform(k_lat, minval=-0.5, maxval=0.5) * half_w,
        heading_offset=jax.random.uniform(k_head, minval=-0.1, maxval=0.1),
        speed=jax.random.uniform(k_speed, minval=cfg.start_speed[0], maxval=cfg.start_speed[1]),
    )


def transition(state: EnvState, action, tracks: Track, params: CarParams, cfg: EnvConfig):
    """One decision, no auto-reset. Returns (state, obs, reward, terminated, truncated, info)."""
    action = jnp.clip(action, ACTION_LOW, ACTION_HIGH)
    car = state.car
    for _ in range(cfg.action_repeat):  # static count: unrolls at trace time
        car = physics.step(car, action, params)

    tid = state.track_id
    length = tracks.length[tid]
    idx = nearest(tracks, tid, state.idx, car, cfg.search_window)
    along, lateral, _ = frame(tracks, tid, idx, car)
    s = (tracks.s[tid, idx] + along) % length
    ds = (s - state.s + 0.5 * length) % length - 0.5 * length  # wrap across the start line

    limit = 0.5 * interp(tracks.width, tracks, tid, s) + cfg.car_half_width
    over_limits = jnp.abs(lateral) > limit
    stuck = jnp.where(car.vx < cfg.stuck_speed, state.stuck + 1, 0)
    decision_dt = cfg.action_repeat * params.dt
    terminated = (jnp.abs(lateral) > limit + cfg.runoff) | (stuck * decision_dt > cfg.stuck_time)
    reward = (
        cfg.progress_scale * ds
        - cfg.offtrack_penalty * over_limits
        - cfg.jerk_penalty * jnp.sum((action - state.prev_action) ** 2)
        - cfg.crash_penalty * terminated
    )

    t = state.t + 1
    truncated = (t >= cfg.max_steps) & ~terminated

    state = EnvState(
        car=car,
        track_id=tid,
        idx=idx,
        s=s,
        progress=state.progress + ds,
        prev_action=action,
        t=t,
        stuck=stuck,
        offtrack=state.offtrack + over_limits,
        ep_return=state.ep_return + reward,
    )
    info = dict(
        speed=car.vx,
        lateral=lateral,
        lateral_frac=lateral / (limit - cfg.car_half_width),  # 1.0 = centre of car on the edge
        over_limits=over_limits,
        laps=state.progress / length,
        offtrack=state.offtrack,
        ep_return=state.ep_return,
    )
    return state, observe(state, tracks, params, cfg), reward, terminated, truncated, info


def step(key, state: EnvState, action, tracks: Track, params: CarParams, cfg: EnvConfig):
    """transition() plus auto-reset inside jit.

    A reset state is computed every step and selected with jnp.where wherever the episode ended.
    Branching (lax.cond) would not save work under vmap, which evaluates both sides anyway.
    info["final_obs"] is the pre-reset observation, needed to bootstrap the value on truncation.
    """
    state, obs, reward, terminated, truncated, info = transition(state, action, tracks, params, cfg)
    done = terminated | truncated
    reset_state, reset_obs = reset(key, tracks, params, cfg)
    state = jax.tree.map(lambda r, s: jnp.where(done, r, s), reset_state, state)
    info = dict(info, final_obs=obs, terminated=terminated, truncated=truncated)
    return state, jnp.where(done, reset_obs, obs), reward, done, info


# --- Evaluation ---------------------------------------------------------------------------------


def rollout(policy, tracks: Track, tid, params: CarParams, cfg: EnvConfig, n_steps: int):
    """Standing start at s = 0 of track tid, no auto-reset, fixed length. After termination the
    car is frozen and `alive` goes false. policy(state, obs) -> action.
    Returns a dict of per-decision arrays (leading axis n_steps)."""
    state, obs = init_at(tracks, tid, 0, params, cfg)

    def body(carry, _):
        state, obs, alive = carry
        action = policy(state, obs)
        nstate, nobs, _, terminated, _, info = transition(state, action, tracks, params, cfg)
        nstate = jax.tree.map(lambda a, b: jnp.where(alive, a, b), nstate, state)
        nobs = jnp.where(alive, nobs, obs)
        out = dict(
            x=nstate.car.x,
            y=nstate.car.y,
            yaw=nstate.car.yaw,
            speed=nstate.car.vx,
            progress=nstate.progress,
            lateral_frac=info["lateral_frac"],
            over_limits=info["over_limits"] & alive,
            alive=alive,
            action=action,
        )
        return (nstate, nobs, alive & ~terminated), out

    _, traj = jax.lax.scan(body, (state, obs, jnp.bool_(True)), None, length=n_steps)
    return traj


def lap_summary(traj: dict, length: float, decision_dt: float) -> list[dict]:
    """Host-side: split a rollout into completed laps. Each lap reports its time and the number of
    separate track-limit excursions (a lap is clean when that is zero)."""
    import numpy as np

    progress = np.asarray(traj["progress"])
    alive = np.asarray(traj["alive"])
    over = np.asarray(traj["over_limits"]).astype(int)
    events = np.diff(over, prepend=0) == 1
    laps, start = [], 0
    for k in range(1, int(progress[alive].max(initial=0) // length) + 1):
        end = int(np.argmax(progress >= k * length))
        laps.append(
            dict(
                lap=k,
                time=(end + 1 - start) * decision_dt if k > 1 else (end + 1) * decision_dt,
                offtrack_events=int(events[start : end + 1].sum()),
            )
        )
        start = end + 1
    return laps
