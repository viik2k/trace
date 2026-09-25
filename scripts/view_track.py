"""M2: visualise a track in Rerun.

uv run python scripts/view_track.py --ref hairpin
uv run python scripts/view_track.py --file data/tracks_train.npz --index 7
Then: uv run rerun track.rrd
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from trace_rl import track, viz


@dataclass
class Args:
    ref: str | None = None  # oval | hairpin | sweeper
    file: Path | None = None
    index: int = 0
    out: Path = Path("track.rrd")


def main(args: Args) -> None:
    if args.ref:
        tr, name = track.reference_tracks()[args.ref], args.ref
    elif args.file:
        tr, name = (
            track.to_numpy(track.load(args.file), args.index),
            f"{args.file.stem}_{args.index}",
        )
    else:
        raise SystemExit("pass --ref or --file")
    viz.start(args.out)
    viz.log_track(f"track/{name}", tr)
    k = np.abs(tr["curvature"])
    print(
        f"{name}: length {tr['length']:.0f} m, {tr['n']} points at {tr['ds']:.3f} m, "
        f"min radius {1 / k.max():.1f} m, width {tr['width'].min():.1f}-{tr['width'].max():.1f} m"
    )
    print(f"wrote {args.out}; open with: uv run rerun {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
