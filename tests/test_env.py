import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trace_rl import env, pure_pursuit, track
from trace_rl.env import EnvConfig
from trace_rl.physics import CarParams

P, CFG = CarParams(), EnvConfig()
# crash_penalty is off by default; the two termination tests switch it on to check where it lands.
CRASH = EnvConfig(crash_penalty=2.0)
DT = CFG.action_repeat * P.dt
REFS = track.reference_tracks()
TRACKS = track.stack(list(REFS.values()))


@pytest.mark.parametrize(
    "car",
    [{}, {"cla": 3.0}, {"load_transfer": 1.0}, {"load_transfer": 1.0, "mu_rear_scale": 1.05}],
)
def test_pure_pursuit_clean_laps_on_reference_tracks(car):
    p = CarParams(**car)
    vprof = jnp.asarray(np.stack([pure_pursuit.speed_profile(t, p) for t in REFS.values()]))
    policy = lambda st, obs: pure_pursuit.act(st, TRACKS, vprof, p)  # noqa: E731
    run = jax.jit(jax.vmap(lambda tid: env.rollout(policy, TRACKS, tid, p, CFG, int(150 / DT))))
    trajs = jax.tree.map(np.asarray, run(jnp.arange(len(REFS))))
    for i, name in enumerate(REFS):
        traj = jax.tree.map(lambda a: a[i], trajs)  # noqa: B023
        laps = env.lap_summary(traj, REFS[name]["length"], DT)
        print(
            f"{name}: "
            + ", ".join(
                f"lap {lap['lap']} {lap['time']:.2f}s off-track {lap['offtrack_events']}"
                for lap in laps
            )
        )
        assert traj["alive"].all(), f"{name}: terminated"
        assert len(laps) >= 2, f"{name}: fewer than 2 laps"
        assert all(lap["offtrack_events"] == 0 for lap in laps), f"{name}: left the track"


def test_observation_has_no_global_position():
    # Rotate and translate the whole world: the observation sequence must not change.
    tr = REFS["hairpin"]
    theta, shift = 1.1, np.array([5000.0, -3000.0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    moved = dict(tr, xy=tr["xy"] @ rot.T + shift, heading=tr["heading"] + theta)
    both = track.stack([tr, moved])
    actions = jax.random.uniform(jax.random.key(1), (200, 3)) * jnp.array([0.2, 0.6, 0.1])

    def obs_seq(tid):
        state, obs = env.init_at(
            both, tid, 10, P, CFG, lateral=1.5, heading_offset=0.05, speed=20.0
        )

        def body(state, a):
            state, obs, *_ = env.transition(state, a, both, P, CFG)
            return state, obs

        return jax.lax.scan(body, state, actions)[1]

    a, b = np.asarray(jax.jit(obs_seq)(0)), np.asarray(jax.jit(obs_seq)(1))
    # float32 at |x| ~ 5 km has ~0.5 mm resolution, so allow a small tolerance.
    np.testing.assert_allclose(a, b, atol=2e-3)


def test_driving_backwards_earns_negative_progress():
    # Guards the circling/reversing exploit: progress is signed arc length, not distance driven.
    state, _ = env.init_at(TRACKS, 0, 0, P, CFG, heading_offset=np.pi, speed=15.0)

    def body(state, _):
        state, _, reward, *_ = env.transition(state, jnp.array([0.0, 0.3, 0.0]), TRACKS, P, CFG)
        return state, reward

    state, rewards = jax.jit(lambda s: jax.lax.scan(body, s, None, length=100))(state)
    print(f"progress after 5 s backwards: {float(state.progress):.1f} m")
    assert float(state.progress) < -50.0
    assert float(rewards.sum()) < 0.0


def test_auto_reset_on_termination():
    # Full lock at speed leaves the track; the env must reset itself inside jit.
    state, _ = env.init_at(TRACKS, 1, 0, P, CRASH, speed=40.0)

    def body(carry, key):
        state = carry
        state, obs, reward, done, info = env.step(
            key, state, jnp.array([1.0, 1.0, 0.0]), TRACKS, P, CRASH
        )
        return state, (done, info["terminated"], info["truncated"], state.t, reward)

    keys = jax.random.split(jax.random.key(0), 200)
    _, (done, term, trunc, t, reward) = jax.tree.map(
        np.asarray, jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)
    )
    assert term.any() and not trunc.any()
    k = int(np.argmax(done))
    assert t[k] == 0 and t[k + 1] == 1  # step counter restarted after the reset
    assert reward[k] < -0.9 * CRASH.crash_penalty < reward[k - 1]  # penalty lands on the crash step


def test_stopping_is_penalised_like_a_crash():
    # Guards the dodge where the car brakes to a stop instead of crashing: stuck terminates with
    # the same penalty.
    state, _ = env.init_at(TRACKS, 0, 0, P, CRASH, speed=0.0)

    def body(state, _):
        state, _, reward, term, *_ = env.transition(
            state, jnp.array([0.0, 0.0, 1.0]), TRACKS, P, CRASH
        )
        return state, (reward, term)

    _, (reward, term) = jax.tree.map(
        np.asarray, jax.jit(lambda s: jax.lax.scan(body, s, None, length=60))(state)
    )
    k = int(np.argmax(term))
    assert term.any() and abs(k * DT - CRASH.stuck_time) < 0.1
    np.testing.assert_allclose(reward[k], -CRASH.crash_penalty, atol=1e-3)
