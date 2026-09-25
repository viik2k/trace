import jax
import jax.numpy as jnp
import numpy as np

from trace_rl.physics import RHO_AIR, CarParams, G, _accelerations, init_state, step

P = CarParams()


def _drive(action, n_steps, params=P):
    """Constant action from rest; returns the trajectory as numpy arrays."""

    def body(s, _):
        s = step(s, action, params)
        return s, s

    _, traj = jax.jit(lambda: jax.lax.scan(body, init_state(), None, length=n_steps))()
    return jax.tree.map(np.asarray, traj)


def test_top_speed_plausible():
    tr = _drive(jnp.array([0.0, 1.0, 0.0]), 60 * 90)
    v_kmh = tr.vx * 3.6
    print(f"top speed {v_kmh[-1]:.1f} km/h, 0-100 {np.argmax(v_kmh > 100) / 60:.2f} s")
    assert 250 < v_kmh[-1] < 310
    assert abs(v_kmh[-1] - v_kmh[-60 * 5]) < 1.0  # plateaued over the last 5 s
    assert np.all(np.abs(tr.y) < 1e-3)  # straight means straight


def test_constant_steer_gives_steady_circle():
    # ~25 m/s at half the grip limit. Speed settles with a time constant of tens of seconds (drag
    # is small), so run 2 minutes and judge the last 10 s.
    tr = _drive(jnp.array([0.1, 0.1, 0.0]), 60 * 120)
    last = slice(-60 * 10, None)
    x, y = tr.x[last], tr.y[last]
    # Algebraic circle fit: x^2 + y^2 + D x + E y + F = 0.
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    D, E, F = np.linalg.lstsq(A, -(x**2 + y**2), rcond=None)[0]
    cx, cy = -D / 2, -E / 2
    radius = np.sqrt(cx**2 + cy**2 - F)
    resid = np.abs(np.hypot(x - cx, y - cy) - radius)
    kin = (P.lf + P.lr) / np.tan(0.1 * P.max_steer)
    print(
        f"radius {radius:.2f} m (kinematic {kin:.1f}), max fit residual {resid.max():.3f} m, "
        f"speed {tr.vx[-1]:.2f} m/s, lateral accel {tr.vx[-1] ** 2 / radius:.2f} m/s^2"
    )
    assert resid.max() < 0.01 * radius
    assert abs(tr.vx[-1] - tr.vx[-600]) / tr.vx[-1] < 0.01  # speed settled
    assert radius > kin  # mild understeer by design


def test_low_speed_radius_matches_kinematic():
    # At low speed tyre slip is negligible, so radius -> wheelbase / tan(steer angle).
    tr = _drive(jnp.array([0.5, 0.03, 0.0]), 60 * 30)
    radius = np.hypot(tr.vx[-1], tr.vy[-1]) / tr.r[-1]
    expected = (P.lf + P.lr) / np.tan(0.5 * P.max_steer)
    print(f"speed {tr.vx[-1]:.2f} m/s, radius {radius:.2f} m, kinematic {expected:.2f} m")
    assert abs(radius - expected) / expected < 0.03


def test_no_nans_at_extreme_inputs():
    n_cars, n_steps = 256, 10_000
    key = jax.random.key(0)
    # Bang-bang inputs, re-drawn every 10 steps, including out-of-range values the step must clip.
    raw = jax.random.choice(key, jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0]), (n_steps // 10, n_cars, 3))
    actions = jnp.repeat(raw, 10, axis=0)

    def body(s, a):
        s = jax.vmap(step, in_axes=(0, 0, None))(s, a, P)
        return s, (s.vx, s.vy, s.r)

    s0 = jax.vmap(lambda v: init_state(vx=v))(jnp.linspace(0.0, 80.0, n_cars))
    final, (vx, vy, r) = jax.jit(lambda: jax.lax.scan(body, s0, actions))()
    vx, vy, r = map(np.asarray, (vx, vy, r))
    for leaf in jax.tree.leaves(final):
        assert np.all(np.isfinite(leaf))
    assert np.all(np.isfinite(vx)) and np.all(np.isfinite(vy)) and np.all(np.isfinite(r))
    print(
        f"max vx {np.abs(vx).max():.1f}, max vy {np.abs(vy).max():.1f}, max r {np.abs(r).max():.2f}"
    )
    assert vx.min() >= 0.0 and vx.max() < 90.0 and np.abs(vy).max() < 90.0


def test_vmap_over_car_params():
    # Params are a pytree, so a batch of different cars steps in one vmapped call.
    masses = jnp.array([1100.0, 1300.0, 1600.0])
    cars = jax.vmap(lambda m: CarParams(mass=m))(masses)
    s0 = jax.vmap(lambda _: init_state())(masses)
    step_all = jax.jit(jax.vmap(step, in_axes=(0, None, 0)))
    s = s0
    for _ in range(120):
        s = step_all(s, jnp.array([0.0, 1.0, 0.0]), cars)
    vx = np.asarray(s.vx)
    print("speed after 2 s by mass:", vx.round(2))
    assert vx[0] > vx[1] > vx[2]  # lighter car accelerates harder


def test_downforce_scales_grip():
    # Pure front slip, no pedals: lateral accel is linear in front Fz, so scales by (mg + L) / mg.
    s = init_state(vx=60.0)
    ay = lambda p: float(_accelerations(s, p.max_steer, 0.0, 0.0, p)[1])  # noqa: E731
    aero = CarParams(cla=3.0)
    expected = 1 + 0.5 * RHO_AIR * aero.cla * 60.0**2 / (aero.mass * G)
    print(f"lateral grip ratio at 60 m/s: {ay(aero) / ay(P):.3f} (expected {expected:.3f})")
    np.testing.assert_allclose(ay(aero) / ay(P), expected, rtol=1e-4)
