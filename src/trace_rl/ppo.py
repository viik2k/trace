"""M4: PPO in one file, PureJaxRL style.

Rollout, GAE and the PPO update all run inside one jitted `train_chunk`: a lax.scan over
`updates_per_chunk` updates, with no host round-trips inside. Between chunks the host logs
metrics, runs a jitted validation rollout on held-out procedural tracks, and checkpoints on the
best result. Reference tracks are never touched here; they are for M5 only.

uv run python -m trace_rl.ppo --smoke          # tiny run, checks the pipeline end to end
uv run python -m trace_rl.ppo --aim-repo aim://<host>:53800
"""

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from trace_rl import env, track
from trace_rl.env import EnvConfig
from trace_rl.physics import CarParams


@dataclass
class Config:
    seed: int = 0
    train_tracks: Path = Path("data/tracks_train.npz")
    val_tracks: Path = Path("data/tracks_val.npz")
    n_envs: int = 4096
    n_steps: int = 64  # decisions per env per update
    total_steps: int = 300_000_000  # decisions summed over all envs
    updates_per_chunk: int = 25  # updates per jitted call; logging/eval/checkpoint happen between
    n_epochs: int = 4
    n_minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    depth: int = 2
    init_log_std: float = -0.5
    val_seconds: float = 300.0  # standing-start rollout length per validation track
    run_dir: Path = Path("runs")
    run_name: str = ""  # default: timestamp
    aim_repo: str = ""  # aim://host:53800 or a local path; empty = stdout only
    smoke: bool = False  # tiny config for a quick CPU check
    env: EnvConfig = field(default_factory=EnvConfig)


SMOKE = dict(
    n_envs=64, n_steps=32, total_steps=64 * 32 * 10, updates_per_chunk=5, n_minibatches=4,
    hidden=64, val_seconds=20.0,
)  # fmt: skip


# --- Policy -------------------------------------------------------------------------------------


class ActorCritic(eqx.Module):
    actor: eqx.nn.MLP
    critic: eqx.nn.MLP
    log_std: jax.Array  # state-independent, per action dimension

    def __init__(self, obs_dim: int, hidden: int, depth: int, init_log_std: float, key):
        ka, kc = jax.random.split(key)
        actor = eqx.nn.MLP(obs_dim, 3, hidden, depth, activation=jnp.tanh, key=ka)
        # Shrink the last layer so the initial mean is ~0 for any observation.
        last = actor.layers[-1]
        self.actor = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), actor,
            (last.weight * 0.01, last.bias * 0.01),
        )  # fmt: skip
        self.critic = eqx.nn.MLP(obs_dim, "scalar", hidden, depth, activation=jnp.tanh, key=kc)
        # Explicit dtype: a weakly-typed leaf changes type after one update and forces a recompile.
        self.log_std = jnp.full(3, init_log_std, dtype=jnp.float32)


def to_env_action(u):
    """Squash a pre-tanh sample to env bounds: steer [-1, 1], throttle and brake [0, 1]."""
    a = jnp.tanh(u)
    return jnp.concatenate([a[..., :1], 0.5 * (a[..., 1:] + 1.0)], axis=-1)


def gaussian_logp(u, mean, log_std):
    # Log-prob of the pre-squash sample u. The tanh correction -sum(log(1 - tanh(u)^2)) depends
    # only on u, which the buffer stores, so it is the same under the old and new policy and
    # cancels exactly in the PPO ratio. It is never needed, so never computed.
    z = (u - mean) / jnp.exp(log_std)
    return jnp.sum(-0.5 * z**2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


# --- Training -----------------------------------------------------------------------------------


class Batch(NamedTuple):
    # NamedTuple is a pytree, so lax.scan can stack one per step into (n_steps, n_envs, ...).
    obs: jax.Array
    u: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    # Episode statistics, meaningful where done.
    ep_return: jax.Array
    ep_laps: jax.Array
    terminated: jax.Array
    # Per-decision statistics.
    over_limits: jax.Array
    speed: jax.Array


def gae(b: Batch, last_value, gamma, lam):
    def body(carry, t):
        next_adv, next_value = carry
        not_done = 1.0 - t.done
        delta = t.reward + gamma * next_value * not_done - t.value
        adv = delta + gamma * lam * not_done * next_adv
        return (adv, t.value), adv

    _, adv = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value), b, reverse=True)
    return adv, adv + b.value


def loss_fn(model: ActorCritic, mb, cfg: Config):
    obs, u, logp_old, adv, ret = mb
    mean = jax.vmap(model.actor)(obs)
    value = jax.vmap(model.critic)(obs)
    ratio = jnp.exp(gaussian_logp(u, mean, model.log_std) - logp_old)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    clipped = jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
    pg_loss = -jnp.minimum(ratio * adv, clipped * adv).mean()
    v_loss = 0.5 * ((value - ret) ** 2).mean()
    # Entropy of the pre-squash Gaussian: the squashed distribution has no closed form.
    entropy = jnp.sum(model.log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
    return loss, dict(
        pg_loss=pg_loss, v_loss=v_loss, entropy=entropy,
        approx_kl=((ratio - 1) - jnp.log(ratio)).mean(),
        clip_frac=(jnp.abs(ratio - 1) > cfg.clip_eps).mean(),
    )  # fmt: skip


def episode_metrics(b: Batch):
    n = jnp.maximum(b.done.sum(), 1)
    return dict(
        episodes=b.done.sum(),
        ep_return=(b.ep_return * b.done).sum() / n,
        lap_completion=((b.ep_laps >= 1.0) & b.done).sum() / n,
        termination_rate=(b.terminated & b.done).sum() / n,  # the rest hit the time limit
        offtrack_rate=b.over_limits.mean(),  # fraction of decisions past track limits
        mean_speed_kmh=b.speed.mean() * 3.6,
    )


def make_train_chunk(cfg: Config, static, optimizer, car: CarParams):
    env_step = jax.vmap(
        lambda k, s, a, tr: env.step(k, s, a, tr, car, cfg.env), in_axes=(0, 0, 0, None)
    )

    def update(runner, tracks):
        params, opt_state, env_state, obs, key = runner
        model = eqx.combine(params, static)

        def rollout_step(carry, _):
            env_state, obs, key = carry
            # PRNG keys are values, not global state: split a fresh subkey for every use and
            # carry the remainder forward. Reusing a key would repeat the same "random" numbers.
            key, k_act, k_env = jax.random.split(key, 3)
            mean = jax.vmap(model.actor)(obs)
            value = jax.vmap(model.critic)(obs)
            u = mean + jnp.exp(model.log_std) * jax.random.normal(k_act, mean.shape)
            env_state, next_obs, reward, done, info = env_step(
                jax.random.split(k_env, cfg.n_envs), env_state, to_env_action(u), tracks
            )
            # A time limit is not a real ending. Bootstrap through it by folding gamma * V(final
            # obs) into the reward, then cut the trace there like any other done.
            v_final = jax.vmap(model.critic)(info["final_obs"])
            reward = reward + cfg.gamma * v_final * info["truncated"]
            b = Batch(
                obs, u, gaussian_logp(u, mean, model.log_std), value, reward, done,
                info["ep_return"], info["laps"], info["terminated"], info["over_limits"],
                info["speed"],
            )  # fmt: skip
            return (env_state, next_obs, key), b

        (env_state, obs, key), b = jax.lax.scan(
            rollout_step, (env_state, obs, key), None, length=cfg.n_steps
        )
        adv, ret = gae(b, jax.vmap(model.critic)(obs), cfg.gamma, cfg.gae_lambda)

        def epoch(carry, _):
            params, opt_state, key = carry
            key, k_perm = jax.random.split(key)
            flat = jax.tree.map(
                lambda a: a.reshape(-1, *a.shape[2:]), (b.obs, b.u, b.logp, adv, ret)
            )
            perm = jax.random.permutation(k_perm, cfg.n_steps * cfg.n_envs)
            mbs = jax.tree.map(lambda a: a[perm].reshape(cfg.n_minibatches, -1, *a.shape[1:]), flat)

            def minibatch(carry, mb):
                params, opt_state = carry
                # Differentiate w.r.t. the array leaves only; `static` holds the non-array parts
                # of the model (activation functions etc.) that jax can't trace.
                grads, aux = jax.grad(
                    lambda p: loss_fn(eqx.combine(p, static), mb, cfg), has_aux=True
                )(params)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), aux

            (params, opt_state), aux = jax.lax.scan(minibatch, (params, opt_state), mbs)
            return (params, opt_state, key), aux

        key, k_epochs = jax.random.split(key)
        (params, opt_state, _), aux = jax.lax.scan(
            epoch, (params, opt_state, k_epochs), None, length=cfg.n_epochs
        )
        metrics = episode_metrics(b) | jax.tree.map(jnp.mean, aux)
        return (params, opt_state, env_state, obs, key), metrics

    def train_chunk(runner, tracks):
        # tracks is an argument, not a closure: a closed-over array is baked into the compiled
        # program as a constant (~130 MB for the training pool).
        return jax.lax.scan(
            lambda r, _: update(r, tracks), runner, None, length=cfg.updates_per_chunk
        )

    # donate_argnums: the old runner buffers are reused for the new one, halving peak memory.
    return jax.jit(train_chunk, donate_argnums=0)


def make_evaluate(cfg: Config, static, car: CarParams):
    decision_dt = cfg.env.action_repeat * car.dt
    n = int(cfg.val_seconds / decision_dt)

    def evaluate(params, tracks):
        """Deterministic policy, standing start, first lap on every track in `tracks`."""
        model = eqx.combine(params, static)
        policy = lambda state, obs: to_env_action(model.actor(obs))  # noqa: E731
        trajs = jax.vmap(lambda tid: env.rollout(policy, tracks, tid, car, cfg.env, n))(
            jnp.arange(tracks.n.shape[0])
        )
        reached = trajs["progress"] >= tracks.length[:, None]
        completed = reached.any(axis=1)
        k = jnp.argmax(reached, axis=1)
        before_line = jnp.arange(n)[None, :] <= k[:, None]
        clean = completed & ~(trajs["over_limits"] & before_line).any(axis=1)
        return dict(completed=completed, clean=clean, lap_time=(k + 1) * decision_dt)

    return jax.jit(evaluate)


def save_checkpoint(path: Path, params, static) -> None:
    eqx.tree_serialise_leaves(path, eqx.combine(params, static))


def load_checkpoint(path: Path) -> tuple[ActorCritic, Config]:
    """Rebuild the model from the run's config.json, then fill in the saved leaves."""
    raw = json.loads((Path(path).parent / "config.json").read_text())
    env_cfg = EnvConfig(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in raw.pop("env").items()}
    )
    cfg = Config(**{k: Path(v) if k in ("train_tracks", "val_tracks", "run_dir") else v
                    for k, v in raw.items()}, env=env_cfg)  # fmt: skip
    template = ActorCritic(cfg.env.obs_dim, cfg.hidden, cfg.depth, cfg.init_log_std,
                           jax.random.key(0))  # fmt: skip
    return eqx.tree_deserialise_leaves(path, template), cfg


def main(cfg: Config) -> None:
    if cfg.smoke:
        cfg = dataclasses.replace(cfg, **SMOKE)
    run_name = cfg.run_name or time.strftime("%Y%m%d-%H%M%S")
    out = cfg.run_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    cfg_json = json.loads(json.dumps(dataclasses.asdict(cfg), default=str))
    (out / "config.json").write_text(json.dumps(cfg_json, indent=2))

    tracks, val_tracks = track.load(cfg.train_tracks), track.load(cfg.val_tracks)
    if cfg.smoke:
        tracks = jax.tree.map(lambda a: a[:16], tracks)
        val_tracks = jax.tree.map(lambda a: a[:4], val_tracks)

    aim_run = None
    if cfg.aim_repo:
        from aim import Run  # heavy import, only when tracking

        aim_run = Run(repo=cfg.aim_repo, experiment="trace-ppo")
        aim_run.name = run_name
        aim_run["hparams"] = cfg_json

    car = CarParams()
    key = jax.random.key(cfg.seed)
    key, k_model, k_env = jax.random.split(key, 3)
    model = ActorCritic(cfg.env.obs_dim, cfg.hidden, cfg.depth, cfg.init_log_std, k_model)
    # partition: array leaves (trained, carried through scan) vs everything else (static).
    params, static = eqx.partition(model, eqx.is_array)

    steps_per_update = cfg.n_envs * cfg.n_steps
    n_chunks = max(math.ceil(cfg.total_steps / (steps_per_update * cfg.updates_per_chunk)), 1)
    n_grad_steps = n_chunks * cfg.updates_per_chunk * cfg.n_epochs * cfg.n_minibatches
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(optax.linear_schedule(cfg.lr, 0.0, n_grad_steps), eps=1e-5),
    )
    env_state, obs = jax.vmap(lambda k: env.reset(k, tracks, car, cfg.env))(
        jax.random.split(k_env, cfg.n_envs)
    )
    runner = (params, optimizer.init(params), env_state, obs, key)
    train_chunk = make_train_chunk(cfg, static, optimizer, car)
    evaluate = make_evaluate(cfg, static, car)

    print(f"run {out}  devices {jax.devices()}  {n_chunks} chunks x {cfg.updates_per_chunk} "
          f"updates x {steps_per_update} decisions")  # fmt: skip
    best = (-1.0, -np.inf)
    for c in range(n_chunks):
        t0 = time.perf_counter()
        runner, metrics = train_chunk(runner, tracks)
        metrics = jax.device_get(metrics)  # one device->host transfer per chunk
        train_time = time.perf_counter() - t0

        step0 = c * cfg.updates_per_chunk * steps_per_update
        if aim_run is not None:
            for u in range(cfg.updates_per_chunk):
                for name, v in metrics.items():
                    aim_run.track(float(v[u]), name=name, step=step0 + (u + 1) * steps_per_update,
                                  context={"subset": "train"})  # fmt: skip

        val = jax.device_get(evaluate(runner[0], val_tracks))
        clean_rate = float(val["clean"].mean())
        mean_time = float(val["lap_time"][val["clean"]].mean()) if val["clean"].any() else np.nan
        step = (c + 1) * cfg.updates_per_chunk * steps_per_update
        if aim_run is not None:
            for name, v in [("clean_lap_rate", clean_rate),
                            ("completion_rate", float(val["completed"].mean())),
                            ("clean_lap_time", mean_time)]:  # fmt: skip
                if np.isfinite(v):
                    aim_run.track(v, name=name, step=step, context={"subset": "val"})

        save_checkpoint(out / "last.eqx", runner[0], static)
        score = (clean_rate, -mean_time if np.isfinite(mean_time) else -np.inf)
        tag = ""
        if score > best:
            best = score
            save_checkpoint(out / "best.eqx", runner[0], static)
            (out / "best.json").write_text(json.dumps(
                dict(step=step, val_clean_lap_rate=clean_rate, val_clean_lap_time=mean_time)
            ))  # fmt: skip
            tag = "  * best"
        last = {k: float(v[-1]) for k, v in metrics.items()}
        print(
            f"[{c + 1}/{n_chunks}] {step / 1e6:7.1f}M steps  "
            f"{cfg.updates_per_chunk * steps_per_update / train_time / 1e3:6.0f}k sps | "
            f"return {last['ep_return']:6.2f}  laps {last['lap_completion']:.2f}  "
            f"offtrack {last['offtrack_rate']:.3f}  term {last['termination_rate']:.2f}  "
            f"{last['mean_speed_kmh']:5.1f} km/h  ent {last['entropy']:5.2f}  "
            f"kl {last['approx_kl']:.1e} | val clean {clean_rate:.2f} "
            f"time {mean_time:6.1f}s{tag}",
            flush=True,
        )
    if aim_run is not None:
        aim_run.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
