import numpy as np
import pytest

from trace_rl import track
from trace_rl.track import GenConfig

REFS = track.reference_tracks()


@pytest.mark.parametrize("name", list(REFS))
def test_reference_tracks_valid(name):
    tr = REFS[name]
    assert track.check(tr, GenConfig()) is None
    print(f"{name}: {tr['length']:.0f} m, min radius {1 / np.abs(tr['curvature']).max():.1f} m")


def _spacing_ok(tr):
    d = np.linalg.norm(np.diff(np.vstack([tr["xy"], tr["xy"][:1]]), axis=0), axis=1)
    return np.allclose(d, tr["ds"], rtol=0.02)  # closed loop, uniform arc spacing


def _curvature_matches_heading(tr):
    dh = np.diff(np.concatenate([tr["heading"], tr["heading"][:1]]))
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    k_mid = 0.5 * (tr["curvature"] + np.roll(tr["curvature"], -1))
    # Finite-difference error peaks where curvature changes fastest (hairpin entry/exit).
    return np.abs(dh / tr["ds"] - k_mid).max() < 0.05 * np.abs(tr["curvature"]).max()


def test_geometry_consistent():
    pool, _ = track.generate_pool(50, seed=7)
    for tr in list(REFS.values()) + pool:
        assert _spacing_ok(tr)
        assert _curvature_matches_heading(tr)
        assert abs(np.abs(tr["curvature"]).sum() * tr["ds"]) > 2 * np.pi - 0.05  # turns >= 360


def test_pool_properties():
    cfg = GenConfig()
    pool, stats = track.generate_pool(200, seed=3)
    assert stats["ok"] == 200
    assert all(track.check(t, cfg) is None for t in pool)
    total_turn = np.array([t["curvature"].sum() * t["ds"] for t in pool])
    assert np.allclose(np.abs(total_turn), 2 * np.pi, atol=0.05)  # simple closed loops
    cw = np.mean(total_turn < 0)
    print(f"clockwise fraction {cw:.2f}")
    assert 0.35 < cw < 0.65


def test_rejects_self_intersection():
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    figure8 = np.stack([400 * np.sin(t), 200 * np.sin(2 * t)], axis=1)
    tr = track._from_control_points(figure8, lambda s, length: np.full_like(s, 12.0))
    assert track.check(tr, GenConfig()) == "overlap"


def test_rejects_close_parallel_sections():
    # Hairpin legs 16 m apart with 12 m width: no self-intersection, but only a 4 m gap.
    pi = np.pi
    pts = np.vstack(
        [
            track._line((0, 0), (300, 0), 6),
            track._arc(300, 8, 8, -pi / 2, pi / 2, 6),
            track._line((300, 16), (0, 16), 6),
            track._arc(0, 100, 84, -pi / 2 - 1e-9, -3 * pi / 2, 6),
        ]
    )
    tr = track._from_control_points(pts, lambda s, length: np.full_like(s, 12.0))
    assert track.check(tr, GenConfig(min_radius=5.0, length=(0.0, 1e5))) == "overlap"


def test_stack_save_load_roundtrip(tmp_path):
    pool = list(REFS.values())
    batch = track.stack(pool)
    assert batch.xy.shape == (3, track.MAX_POINTS, 2)
    assert batch.curvature.shape == (3, track.MAX_POINTS)
    assert list(np.asarray(batch.n)) == [t["n"] for t in pool]
    track.save(tmp_path / "t.npz", pool)
    back = track.to_numpy(track.load(tmp_path / "t.npz"), 1)
    np.testing.assert_allclose(back["xy"], REFS["hairpin"]["xy"], rtol=1e-6)
    assert back["n"] == REFS["hairpin"]["n"]
