"""Dump replay.json for the web viewer (arche/trace-viewer): policy and pure pursuit on one track.

uv run python scripts/replay.py --ckpt runs/<run>/best.eqx --ref sweeper --out replay.json
uv run python scripts/replay.py --ckpt runs/<run>/best.eqx --file data/tracks_val.npz --index 3

Schema is documented in the trace-viewer README. Rollouts are standing-start and frozen on crash.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from trace_rl import env, ppo, pure_pursuit, track


@dataclass
class Args:
    ckpt: Path
    ref: str | None = None  # oval | hairpin | sweeper
    file: Path | None = None
    index: int = 0
    seconds: float = 150.0
    out: Path = Path("replay.json")


def main(args: Args) -> None:
    model, cfg = ppo.load_checkpoint(args.ckpt)
    car = cfg.car
    dt = cfg.env.action_repeat * car.dt
    if args.ref:
        tr, name = track.reference_tracks()[args.ref], args.ref
    elif args.file:
        tr = track.to_numpy(track.load(args.file), args.index)
        name = f"{args.file.stem}_{args.index}"
    else:
        raise SystemExit("pass --ref or --file")
    tracks = track.stack([tr])
    vprof = jnp.asarray(pure_pursuit.speed_profile(tr, car))[None]
    drivers = {
        "policy": lambda st, obs: ppo.to_env_action(model.actor(obs)),
        "pure_pursuit": lambda st, obs: pure_pursuit.act(st, tracks, vprof, car),
    }
    n = int(args.seconds / dt)
    cars = {}
    for label, policy in drivers.items():
        traj = jax.tree.map(
            np.asarray, jax.jit(lambda p=policy: env.rollout(p, tracks, 0, car, cfg.env, n))()
        )
        m = int(traj["alive"].sum())  # frames before the crash freeze
        over = traj["over_limits"].astype(int)
        cars[label] = dict(
            x=np.round(traj["x"][:m], 2).tolist(),
            y=np.round(traj["y"][:m], 2).tolist(),
            yaw=np.round(traj["yaw"][:m], 3).tolist(),
            speed=np.round(traj["speed"][:m], 2).tolist(),
            over=over[:m].tolist(),
            crashed=m < n,
            laps=env.lap_summary(traj, tr["length"], dt),
            offtrack_events=int((np.diff(over[:m], prepend=0) == 1).sum()),
        )
    out = dict(
        meta=dict(
            name=name,
            dt=dt,
            length=tr["length"],
            car=dict(length=4.6, width=1.9),  # ponytail: drawn size only, not read from CarParams
        ),
        track={k: np.round(tr[k], 3).tolist() for k in ("xy", "curvature", "width", "s")},
        cars=cars,
    )
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")
    for label, c in cars.items():
        print(label, "frames", len(c["x"]), "crashed", c["crashed"], "laps", c["laps"])


if __name__ == "__main__":
    main(tyro.cli(Args))
