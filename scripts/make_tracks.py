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
    # Share of the train pool swapped for tracks with a corner tighter than tight_radius. The
    # natural pool has few hairpins (p5 min radius ~13 m) and every policy crashed the hairpin.
    tight_frac: float = 0.0
    tight_radius: float = 16.0


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
        if name == "train" and args.tight_frac > 0:
            # ponytail: rejection by oversampling; ~4x the pool gives enough tight tracks at 16 m
            extra, _ = track.generate_pool(4 * n, seed + 100)
            tight = [t for t in extra if np.abs(t["curvature"]).max() > 1 / args.tight_radius]
            tight = tight[: int(n * args.tight_frac)]
            pool = tight + pool[: n - len(tight)]
        describe(name, pool, stats)
        path = args.out / f"tracks_{name}.npz"
        track.save(path, pool)
        print(f"  wrote {path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(tyro.cli(Args))
