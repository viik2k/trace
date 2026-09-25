"""M2: generate the training and validation track pools.

uv run python scripts/make_tracks.py
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from trace_rl import track


@dataclass
class Args:
    n_train: int = 2048
    n_val: int = 64  # held-out procedural tracks for checkpoint selection
    seed: int = 0
    out: Path = Path("data")


def describe(name: str, pool: list[dict], stats: dict) -> None:
    length = np.array([t["length"] for t in pool])
    min_r = np.array([1 / np.abs(t["curvature"]).max() for t in pool])
    cw = np.mean([t["curvature"].sum() < 0 for t in pool])
    tried = sum(stats.values())
    print(
        f"{name}: {len(pool)} tracks, {tried} attempts ({100 * len(pool) / tried:.0f}% accepted), "
        f"rejected { ({k: v for k, v in stats.items() if k != 'ok'}) }\n"
        f"  length m   p5/p50/p95 {np.percentile(length, [5, 50, 95]).round(0)}\n"
        f"  min radius p5/p50/p95 {np.percentile(min_r, [5, 50, 95]).round(1)}\n"
        f"  clockwise {cw:.2f}"
    )


def main(args: Args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    for name, n, seed in [("train", args.n_train, args.seed), ("val", args.n_val, args.seed + 1)]:
        t0 = time.time()
        pool, stats = track.generate_pool(n, seed)
        describe(name, pool, stats)
        path = args.out / f"tracks_{name}.npz"
        track.save(path, pool)
        print(f"  wrote {path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(tyro.cli(Args))
