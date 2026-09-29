"""Robustness on fresh procedural tracks: per checkpoint, how many tracks it goes off or crashes on.

Deterministic policy, standing start, a fixed time on each track (not just the first lap). The
reference tracks and the 64 val tracks saturate (val clean 1.0 on most seeds), this does not.

uv run python scripts/fresh_eval.py runs/r8-hp50-s0/best.eqx runs/r8-hp50-s1/best.eqx
"""

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import tyro

from trace_rl import env, ppo, track


@dataclass
class Args:
    ckpts: tyro.conf.Positional[tuple[Path, ...]]
    n: int = 256
    seed: int = 4242  # never used for a train or val pool
    hairpin_prob: float = 0.25
    seconds: float = 150.0


def run(ckpt: Path, tracks: track.Track, seconds: float):
    """Per track: (went off, crashed) for one checkpoint."""
    model, cfg = ppo.load_checkpoint(ckpt)
    n_steps = int(seconds / (cfg.env.action_repeat * cfg.car.dt))
    policy = lambda state, obs: ppo.to_env_action(model.actor(obs))  # noqa: E731
    rollout = jax.vmap(lambda tid: env.rollout(policy, tracks, tid, cfg.car, cfg.env, n_steps))
    traj = jax.device_get(eqx.filter_jit(rollout)(np.arange(tracks.n.shape[0])))
    return traj["over_limits"].any(axis=1), ~traj["alive"][:, -1]


def main(args: Args) -> None:
    gen = track.GenConfig(hairpin_prob=args.hairpin_prob)
    tracks = track.stack(track.generate_pool(args.n, args.seed, gen)[0])
    totals = np.zeros(2, int)
    for ckpt in args.ckpts:
        off, crash = run(ckpt, tracks, args.seconds)
        totals += off.sum(), crash.sum()
        print(f"{ckpt}  tracks with offs {off.sum():3d}  crashes {crash.sum():3d}", flush=True)
    print(f"total  tracks with offs {totals[0]}  crashes {totals[1]}  over {len(args.ckpts)} ckpts")


if __name__ == "__main__":
    main(tyro.cli(Args))
