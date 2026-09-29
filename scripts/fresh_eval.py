"""Robustness on fresh procedural tracks: per checkpoint, how many tracks it goes off or crashes on.

Deterministic policy, standing start, a fixed time on each track (not just the first lap). The
reference tracks and the 64 val tracks saturate (val clean 1.0 on most seeds), this does not.

uv run python scripts/fresh_eval.py runs/r8-hp50-s0/best.eqx runs/r8-hp50-s1/best.eqx
uv run python scripts/fresh_eval.py runs/r8-hp50-s0/last.eqx --hairpin-prob 1.0 --corners
"""

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import tyro

from trace_rl import env, ppo, pure_pursuit, track
from trace_rl.physics import G


@dataclass
class Args:
    ckpts: tyro.conf.Positional[tuple[Path, ...]]
    n: int = 256
    seed: int = 4242  # never used for a train or val pool
    hairpin_prob: float = 0.25
    seconds: float = 150.0
    tracks: Path | None = None  # a saved pool instead (first n), e.g. a train pool
    corners: bool = False  # per off event: corner, side, entry/apex speed vs the grip limit


# The grip limit: full friction for cornering and braking, not pure pursuit's 0.8 / 0.6 margins.
LIMIT = pure_pursuit.PPConfig(grip_frac=1.0, brake_frac=1.0)


def run(ckpt: Path, tracks: track.Track, seconds: float):
    """Standing-start rollouts of one checkpoint on every track, as numpy (K, n_steps) arrays."""
    model, cfg = ppo.load_checkpoint(ckpt)
    n_steps = int(seconds / (cfg.env.action_repeat * cfg.car.dt))
    policy = lambda state, obs: ppo.to_env_action(model.actor(obs))  # noqa: E731
    rollout = jax.vmap(lambda tid: env.rollout(policy, tracks, tid, cfg.car, cfg.env, n_steps))
    return jax.device_get(eqx.filter_jit(rollout)(np.arange(tracks.n.shape[0]))), cfg


def speed_at(traj_i: dict, e: int, pos: float, length: float) -> float:
    """Car speed at track position pos (m) on the pass nearest decision e."""
    prog = traj_i["progress"]
    target = prog[e] + (pos - prog[e] + length / 2) % length - length / 2
    ok = 0 <= target <= prog.max()
    return float(traj_i["speed"][np.argmax(prog >= target)]) if ok else np.nan


def corner_log(traj, tracks: track.Track, cfg) -> list[dict]:
    """One row per off-track excursion. The corner is the stretch around the largest curvature
    within 40 m of the excursion where |curvature| stays above half its peak; entry is its start,
    apex its peak, turn its heading change (deg: a hairpin ~180, a fillet ~90). Inside = the side
    the corner turns towards. The speed limits come from the centreline radius, so they understate
    what a racing line allows in fast corners; `grip` (peak centripetal acceleration over mu g in
    the 2 s before the excursion) says whether the car was actually at the limit."""
    dt = cfg.env.action_repeat * cfg.car.dt
    d = np.diff(np.stack([traj["x"], traj["y"]], axis=-1), axis=1)  # displacement per decision
    course = np.arctan2(d[..., 1], d[..., 0])  # direction of travel, not yaw: slides don't count
    turn = (np.diff(course, axis=1) + np.pi) % (2 * np.pi) - np.pi
    a_lat = np.linalg.norm(d[:, 1:], axis=-1) / dt * np.abs(turn) / dt  # v * course rate
    grip = np.pad(a_lat, ((0, 0), (2, 0))) / (cfg.car.mu * G)
    rows = []
    for i in np.flatnonzero(traj["over_limits"].any(axis=1)):
        tr = track.to_numpy(tracks, i)
        k, n, ds, length = tr["curvature"], tr["n"], tr["ds"], tr["length"]
        vlim = pure_pursuit.speed_profile(tr, cfg.car, LIMIT)[:n]
        t = {key: v[i] for key, v in traj.items()}
        over, dead = t["over_limits"], np.flatnonzero(~t["alive"])
        for e in np.flatnonzero(np.diff(over.astype(int), prepend=0) == 1):
            j, w = int(t["progress"][e] % length / ds), int(40 / ds)
            win = (j + np.arange(-w, w + 1)) % n
            apex = win[np.argmax(np.abs(k[win]))]
            a = apex  # walk back to the corner entry; the cap stops it on constant-radius loops
            while np.abs(k[(a - 1) % n]) >= 0.5 * np.abs(k[apex]) and (apex - a) * ds < 300:
                a -= 1
            b = apex  # and forward to its exit
            while np.abs(k[(b + 1) % n]) >= 0.5 * np.abs(k[apex]) and (b - apex) * ds < 300:
                b += 1
            turn = np.degrees(np.abs(k[np.arange(a, b + 1) % n].sum() * ds))
            a %= n
            end = e + int(np.argmin(over[e:])) if not over[e:].all() else len(over)
            inside = np.sign(t["lateral_frac"][e]) == np.sign(k[apex])
            rows.append(
                dict(
                    track=int(i),
                    s=int(t["progress"][e] % length),
                    radius=1 / abs(k[apex]),
                    turn=turn,
                    side="inside" if inside else "outside",
                    decisions=end - e,
                    crash=bool(len(dead) and dead[0] <= end + 1),
                    entry=speed_at(t, e, a * ds, length),
                    entry_limit=vlim[a],
                    apex=speed_at(t, e, apex * ds, length),
                    apex_limit=vlim[apex],
                    grip=float(grip[i, max(e - int(2 / dt), 0) : e + 1].max()),
                )
            )
    return rows


def print_corners(rows: list[dict]) -> None:
    for r in rows:
        er, ar = r["entry"] / r["entry_limit"], r["apex"] / r["apex_limit"]
        print(
            f"  track {r['track']:3d}  s {r['s']:5d}  R {r['radius']:5.1f}  turn {r['turn']:3.0f}  "
            f"{r['side']:7s} "
            f"{r['decisions']:3d} dec  entry {r['entry']:4.1f}/{r['entry_limit']:4.1f} m/s "
            f"({er:.2f})  apex {r['apex']:4.1f}/{r['apex_limit']:4.1f} ({ar:.2f})  "
            f"grip {r['grip']:.2f}" + ("  crash" if r["crash"] else "")
        )
    for side in ("inside", "outside"):
        sel = [r for r in rows if r["side"] == side]
        if sel:
            er = np.array([r["entry"] / r["entry_limit"] for r in sel])
            ar = np.array([r["apex"] / r["apex_limit"] for r in sel])
            gr = np.array([r["grip"] for r in sel])
            print(
                f"  {side}: {len(sel)} offs, median speed/centreline limit entry "
                f"{np.nanmedian(er):.2f} apex {np.nanmedian(ar):.2f}, median grip "
                f"{np.median(gr):.2f}, near the grip limit (>= 0.9) {np.sum(gr >= 0.9)}"
            )


def main(args: Args) -> None:
    if args.tracks:
        tracks = jax.tree.map(lambda a: a[: args.n], track.load(args.tracks))
    else:
        gen = track.GenConfig(hairpin_prob=args.hairpin_prob)
        tracks = track.stack(track.generate_pool(args.n, args.seed, gen)[0])
    totals = np.zeros(2, int)
    for ckpt in args.ckpts:
        traj, cfg = run(ckpt, tracks, args.seconds)
        off, crash = traj["over_limits"].any(axis=1), ~traj["alive"][:, -1]
        rate = traj["over_limits"].sum() / traj["alive"].sum()  # share of driven decisions
        totals += off.sum(), crash.sum()
        print(
            f"{ckpt}  tracks with offs {off.sum():3d}  crashes {crash.sum():3d}  "
            f"off-track {100 * rate:.2f}% of decisions",
            flush=True,
        )
        if args.corners:
            print_corners(corner_log(traj, tracks, cfg))
    print(f"total  tracks with offs {totals[0]}  crashes {totals[1]}  over {len(args.ckpts)} ckpts")


if __name__ == "__main__":
    main(tyro.cli(Args))
