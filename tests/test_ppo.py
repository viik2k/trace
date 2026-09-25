import jax
import jax.numpy as jnp
import numpy as np

from trace_rl import ppo, track


def test_gae_matches_reference_loop():
    T, N, gamma, lam = 12, 5, 0.99, 0.95
    k = jax.random.split(jax.random.key(0), 3)
    reward = jax.random.normal(k[0], (T, N))
    value = jax.random.normal(k[1], (T, N))
    done = jax.random.bernoulli(k[2], 0.2, (T, N)).astype(jnp.float32)
    last_value = jnp.ones(N)
    b = ppo.Batch(*(jnp.zeros((T, N)),) * 3, value, reward, done, *(jnp.zeros((T, N)),) * 5)
    adv, ret = ppo.gae(b, last_value, gamma, lam)

    r, v, d = map(np.asarray, (reward, value, done))
    ref = np.zeros((T, N))
    next_adv, next_v = np.zeros(N), np.asarray(last_value)
    for t in reversed(range(T)):
        delta = r[t] + gamma * next_v * (1 - d[t]) - v[t]
        next_adv = delta + gamma * lam * (1 - d[t]) * next_adv
        ref[t], next_v = next_adv, v[t]
    np.testing.assert_allclose(adv, ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, ref + v, rtol=1e-5, atol=1e-5)


def test_action_squashing_bounds():
    u = jnp.array([[-50.0, -50.0, -50.0], [0.0, 0.0, 0.0], [50.0, 50.0, 50.0]])
    a = np.asarray(ppo.to_env_action(u))
    np.testing.assert_allclose(a, [[-1, 0, 0], [0, 0.5, 0.5], [1, 1, 1]], atol=1e-6)


def test_smoke_training_and_checkpoint_roundtrip(tmp_path):
    train, _ = track.generate_pool(8, seed=0)
    val, _ = track.generate_pool(2, seed=1)
    track.save(tmp_path / "train.npz", train)
    track.save(tmp_path / "val.npz", val)
    cfg = ppo.Config(
        smoke=True, train_tracks=tmp_path / "train.npz", val_tracks=tmp_path / "val.npz",
        run_dir=tmp_path, run_name="smoke",
    )  # fmt: skip
    ppo.main(cfg)
    for f in ("config.json", "best.eqx", "last.eqx", "best.json"):
        assert (tmp_path / "smoke" / f).exists()

    model, loaded = ppo.load_checkpoint(tmp_path / "smoke" / "best.eqx")
    assert loaded.env == cfg.env and loaded.hidden == ppo.SMOKE["hidden"]
    a = np.asarray(ppo.to_env_action(model.actor(jnp.zeros(loaded.env.obs_dim))))
    assert a.shape == (3,) and np.all(np.isfinite(a))
