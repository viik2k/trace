"""M2: procedural closed-loop tracks.

Generation is offline NumPy/SciPy. The result is uniform-arc-length arrays padded to a fixed
size so a pool of tracks stacks into one batched pytree for JAX.

Uniform spacing is the key design choice: arc length s maps to index s / ds with no search,
so lookahead sampling inside jit is just a gather.
"""

from dataclasses import dataclass, fields

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

DS = 2.0  # target centreline spacing, m
MAX_POINTS = 2560  # padded size; covers MAX_LENGTH / DS


class Track(eqx.Module):
    """One track, or a batch of K tracks when every field has a leading K axis."""

    xy: jax.Array  # (M, 2) centreline, m
    heading: jax.Array  # (M,) tangent direction, rad
    curvature: jax.Array  # (M,) 1/m, positive = turning left
    width: jax.Array  # (M,) full width, m
    s: jax.Array  # (M,) arc length at each point, m
    n: jax.Array  # () int32 number of valid points; indices wrap modulo n
    ds: jax.Array  # () spacing, m (length / n, so slightly off DS per track)
    length: jax.Array  # () lap length, m


@dataclass(frozen=True)
class GenConfig:
    n_ctrl: tuple[int, int] = (8, 16)
    base_radius: tuple[float, float] = (150.0, 500.0)
    radius_jitter: tuple[float, float] = (0.35, 1.0)  # control point radius as fraction of base
    aspect: tuple[float, float] = (0.5, 2.5)  # x stretch; elongated shapes give long straights
    width: tuple[float, float] = (10.0, 16.0)
    length: tuple[float, float] = (1200.0, 5000.0)
    min_radius: float = 12.0  # tightest corner allowed, m
    margin: float = 6.0  # min gap between the edges of non-adjacent sections, m


def _from_control_points(ctrl: np.ndarray, width_fn) -> dict:
    """Periodic cubic spline through control points, resampled at uniform arc length."""
    closed = np.vstack([ctrl, ctrl[:1]])
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    spline = CubicSpline(t, closed, bc_type="periodic", axis=0)

    t_dense = np.linspace(0.0, t[-1], 200 * len(t))
    seg = np.linalg.norm(np.diff(spline(t_dense), axis=0), axis=1)
    s_dense = np.concatenate([[0.0], np.cumsum(seg)])
    length = s_dense[-1]
    n = max(int(round(length / DS)), 3)
    s = np.arange(n) * (length / n)
    tu = np.interp(s, s_dense, t_dense)

    xy = spline(tu)
    d1, d2 = spline(tu, 1), spline(tu, 2)
    heading = np.arctan2(d1[:, 1], d1[:, 0])
    curvature = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3
    return dict(
        xy=xy,
        heading=heading,
        curvature=curvature,
        width=width_fn(s, length),
        s=s,
        n=n,
        ds=length / n,
        length=length,
    )


def check(tr: dict, cfg: GenConfig) -> str | None:
    """Returns a rejection reason, or None if the track is valid."""
    if not cfg.length[0] <= tr["length"] <= cfg.length[1]:
        return "length"
    if np.abs(tr["curvature"]).max() > 1.0 / cfg.min_radius:
        return "curvature"
    # Any two points far apart along the track must also be far apart in space. This is strictly
    # stronger than a self-intersection test (a crossing puts two such points within DS of each
    # other), and it also stops the car reaching another section before it is terminated.
    xy, w = tr["xy"], tr["width"]
    pairs = cKDTree(xy).query_pairs(r=w.max() + cfg.margin, output_type="ndarray")
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        arc = np.abs(tr["s"][i] - tr["s"][j])
        arc = np.minimum(arc, tr["length"] - arc)
        gap = np.linalg.norm(xy[i] - xy[j], axis=1) - 0.5 * (w[i] + w[j])
        if np.any((arc > 3 * (w.max() + cfg.margin)) & (gap < cfg.margin)):
            return "overlap"
    return None


def generate(rng: np.random.Generator, cfg: GenConfig = GenConfig()) -> tuple[dict | None, str]:
    """One attempt. Returns (track, "ok") or (None, rejection reason)."""
    n = rng.integers(cfg.n_ctrl[0], cfg.n_ctrl[1] + 1)
    ang = (np.arange(n) + rng.uniform(-0.35, 0.35, n)) * 2 * np.pi / n
    rad = rng.uniform(*cfg.base_radius) * rng.uniform(*cfg.radius_jitter, n)
    ctrl = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
    ctrl *= [rng.uniform(*cfg.aspect), 1.0]
    if rng.random() < 0.5:  # half the tracks run clockwise so left and right turns are balanced
        ctrl = ctrl[::-1]

    w_mean = rng.uniform(cfg.width[0] + 1, cfg.width[1] - 1)
    amp, phase = rng.uniform(-1, 1, 3), rng.uniform(0, 2 * np.pi, 3)

    def width_fn(s, length):
        k = np.arange(1, 4)[:, None]
        w = w_mean + (amp[:, None] * np.cos(2 * np.pi * k * s / length + phase[:, None])).sum(0)
        return np.clip(w, *cfg.width)

    tr = _from_control_points(ctrl, width_fn)
    reason = check(tr, cfg)
    return (None, reason) if reason else (tr, "ok")


def generate_pool(n: int, seed: int, cfg: GenConfig = GenConfig()) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    tracks, stats = [], {"ok": 0, "length": 0, "curvature": 0, "overlap": 0}
    while len(tracks) < n:
        tr, reason = generate(rng, cfg)
        stats[reason] += 1
        if tr is not None:
            tracks.append(tr)
    return tracks, stats


# --- Hand-made reference tracks, held out from training -----------------------------------------


def _arc(cx, cy, r, a0, a1, step=10.0):
    k = max(int(abs(a1 - a0) * r / step), 2)
    a = np.linspace(a0, a1, k, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def _line(p0, p1, step=40.0):
    k = max(int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / step), 1)
    return np.linspace(p0, p1, k, endpoint=False)


def reference_tracks() -> dict[str, dict]:
    const = lambda w: lambda s, length: np.full_like(s, w)  # noqa: E731
    pi = np.pi

    # Oval: 700 m straights, 120 m radius ends.
    oval = np.vstack(
        [
            _line((-350, -120), (350, -120)),
            _arc(350, 0, 120, -pi / 2, pi / 2, step=30),
            _line((350, 120), (-350, 120)),
            _arc(-350, 0, 120, pi / 2, 3 * pi / 2, step=30),
        ]
    )

    # Hairpins: accordion of four legs joined by three 15 m hairpins and a 45 m return loop.
    # Uniform 6 m control spacing: uneven spacing makes the spline overshoot at arc entry/exit.
    hairpin = np.vstack(
        [
            _line((0, 0), (400, 0), 6),
            _arc(400, 15, 15, -pi / 2, pi / 2, 6),
            _line((400, 30), (100, 30), 6),
            _arc(100, 45, 15, -pi / 2, -3 * pi / 2, 6),
            _line((100, 60), (400, 60), 6),
            _arc(400, 75, 15, -pi / 2, pi / 2, 6),
            _line((400, 90), (-100, 90), 6),
            _arc(-100, 45, 45, pi / 2, 3 * pi / 2, 6),
            _line((-100, 0), (0, 0), 6),
        ]
    )

    # Fast sweeper: lobed loop with S-bends, tightest corner ~90 m radius.
    th = np.linspace(0, 2 * pi, 48, endpoint=False)
    rr = 450 * (1 + 0.3 * np.cos(2 * th) + 0.12 * np.sin(3 * th + 0.7))
    sweeper = np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)

    return {
        "oval": _from_control_points(oval, const(15.0)),
        "hairpin": _from_control_points(hairpin, const(12.0)),
        "sweeper": _from_control_points(sweeper, const(12.0)),
    }


# --- Batching and IO ----------------------------------------------------------------------------


def stack(tracks: list[dict], m: int = MAX_POINTS) -> Track:
    """Pad each track to m points and stack into one batched Track."""

    def pad(a):
        out = np.zeros((m, *a.shape[1:]), np.float32)
        out[: len(a)] = a
        return out

    fields = ("xy", "heading", "curvature", "width", "s")
    arrays = {f: np.stack([pad(t[f]) for t in tracks]) for f in fields}
    for t in tracks:
        assert t["n"] <= m, f"track has {t['n']} points, max is {m}"
    return Track(
        **{f: jnp.asarray(a) for f, a in arrays.items()},
        n=jnp.asarray([t["n"] for t in tracks], jnp.int32),
        ds=jnp.asarray([t["ds"] for t in tracks], jnp.float32),
        length=jnp.asarray([t["length"] for t in tracks], jnp.float32),
    )


def save(path, tracks: list[dict]) -> None:
    batch = stack(tracks)
    np.savez_compressed(path, **{f.name: np.asarray(getattr(batch, f.name)) for f in fields(batch)})


def load(path) -> Track:
    with np.load(path) as f:
        return Track(**{k: jnp.asarray(f[k]) for k in f.files})


def unbatch(tracks: Track, i: int) -> Track:
    return jax.tree.map(lambda a: a[i], tracks)


def to_numpy(tracks: Track, i: int) -> dict:
    """Track i of a batch as an unpadded numpy dict (the offline format)."""
    n = int(tracks.n[i])
    return dict(
        xy=np.asarray(tracks.xy[i, :n]),
        heading=np.asarray(tracks.heading[i, :n]),
        curvature=np.asarray(tracks.curvature[i, :n]),
        width=np.asarray(tracks.width[i, :n]),
        s=np.asarray(tracks.s[i, :n]),
        n=n,
        ds=float(tracks.ds[i]),
        length=float(tracks.length[i]),
    )
