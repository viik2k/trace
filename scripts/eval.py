"""M5: evaluate a checkpoint on the held-out reference tracks against pure pursuit.

Standing start, then keep driving; lap 2 (the first flying lap) is the one compared.
Also writes a Rerun replay of the policy's best and worst tracks with the baseline overlaid.

uv run python scripts/eval.py --ckpt runs/<run>/best.eqx
uv run rerun eval.rrd
"""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from trace_rl import env, ppo, pure_pursuit, track, viz
from trace_rl.physics import CarParams


@dataclass
class Args:
    ckpt: Path
    seconds: float = 240.0  # per track: standing-start lap plus at least one flying lap
    out: Path = Path("eval.rrd")


def summarise(traj: dict, length: float, dt: float) -> dict:
    alive = traj["alive"]
    laps = env.lap_summary(traj, length, dt)
    steer = traj["action"][alive, 0]
    dsteer = np.diff(steer)
    moves = dsteer[np.abs(dsteer) > 0.02]  # ignore micro-jitter so reversals mean weaving
    over = traj["over_limits"].astype(int)
    return dict(
        laps=laps,
        terminated=not bool(alive[-1]),
        offtrack_events=int((np.diff(over, prepend=0) == 1).sum()),
        flying_lap=laps[1]["time"] if len(laps) > 1 else None,
        # Style diagnostics that expose common reward hacks: steering oscillation and riding the
        # edge of the track.
        steer_reversals_per_s=np.sum(np.diff(np.sign(moves)) != 0) / (alive.sum() * dt),
        mean_abs_dsteer=float(np.abs(dsteer).mean()),
        edge_time_frac=float(np.mean(np.abs(traj["lateral_frac"][alive]) > 0.9)),
    )


def main(args: Args) -> None:
    model, cfg = ppo.load_checkpoint(args.ckpt)
    car = CarParams()
    dt = cfg.env.action_repeat * car.dt
    refs = track.reference_tracks()
    names = list(refs)
    tracks = track.stack(list(refs.values()))
    vprof = jnp.asarray(np.stack([pure_pursuit.speed_profile(t, car) for t in refs.values()]))
    drivers = {
        "policy": lambda st, obs: ppo.to_env_action(model.actor(obs)),
        "pure_pursuit": lambda st, obs: pure_pursuit.act(st, tracks, vprof, car),
    }

    results = {}
    n = int(args.seconds / dt)
    for driver, policy in drivers.items():
        rollout = jax.vmap(lambda tid, p=policy: env.rollout(p, tracks, tid, car, cfg.env, n))
        trajs = jax.tree.map(np.asarray, jax.jit(rollout)(jnp.arange(len(names))))
        results[driver] = [jax.tree.map(lambda a, i=i: a[i], trajs) for i in range(len(names))]

    rows, all_clean, all_faster = [], True, True
    for i, name in enumerate(names):
        s = {d: summarise(results[d][i], refs[name]["length"], dt) for d in drivers}
        pol, base = s["policy"], s["pure_pursuit"]
        clean = not pol["terminated"] and len(pol["laps"]) >= 2 and pol["offtrack_events"] == 0
        faster = clean and base["flying_lap"] is not None and pol["flying_lap"] < base["flying_lap"]
        all_clean &= clean
        all_faster &= faster
        rows.append((name, s, clean, faster))

    fmt = lambda t: f"{t:7.2f}s" if t is not None else "      --"  # noqa: E731
    print(f"{'track':8s} {'driver':12s} {'laps':>4s} {'flying':>8s} {'offtrk':>6s} {'term':>5s} "
          f"{'rev/s':>6s} {'|dsteer|':>8s} {'edge%':>6s}")  # fmt: skip
    for name, s, clean, faster in rows:
        for d, r in s.items():
            print(f"{name:8s} {d:12s} {len(r['laps']):4d} {fmt(r['flying_lap'])} "
                  f"{r['offtrack_events']:6d} {str(r['terminated']):>5s} "
                  f"{r['steer_reversals_per_s']:6.2f} {r['mean_abs_dsteer']:8.3f} "
                  f"{100 * r['edge_time_frac']:5.1f}%")  # fmt: skip
        print(f"{'':8s} -> clean laps: {clean}, beats baseline: {faster}")
    print(f"\nPhase 1: clean laps on all held-out tracks: {all_clean}; "
          f"beats pure pursuit on every flying lap: {all_faster}")  # fmt: skip

    def ratio(row):  # policy / baseline flying lap; a failed track ranks worst
        _, s, clean, _ = row
        if not clean or s["pure_pursuit"]["flying_lap"] is None:
            return np.inf
        return s["policy"]["flying_lap"] / s["pure_pursuit"]["flying_lap"]

    order = sorted(range(len(rows)), key=lambda i: ratio(rows[i]))
    viz.start(args.out)
    for label, i in [("best", order[0]), ("worst", order[-1])]:
        prefix = f"{label}_{names[i]}"
        viz.log_track(f"{prefix}/track", refs[names[i]])
        viz.log_trajectory(f"{prefix}/policy", results["policy"][i], dt, (255, 200, 0))
        viz.log_trajectory(f"{prefix}/pure_pursuit", results["pure_pursuit"][i], dt, (0, 200, 255))
    print(f"replays: best={names[order[0]]} worst={names[order[-1]]} -> {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
