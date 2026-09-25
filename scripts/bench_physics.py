"""M1 benchmark: physics steps per second for a vmapped batch of cars.

uv run python scripts/bench_physics.py --n-cars 4096
"""

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tyro

from trace_rl.physics import CarParams, init_state, step


@dataclass
class Args:
    n_cars: int = 4096
    n_steps: int = 1000  # physics steps per timed call
    repeats: int = 5


def main(args: Args) -> None:
    params = CarParams()
    key = jax.random.key(0)
    actions = jax.random.uniform(key, (args.n_steps, args.n_cars, 3), minval=-1.0, maxval=1.0)
    actions = actions.at[..., 1:].set(jnp.abs(actions[..., 1:]))

    @jax.jit
    def run(s, actions):
        # The whole rollout is one jitted scan: one dispatch per call, no host round-trips.
        def body(s, a):
            return jax.vmap(step, in_axes=(0, 0, None))(s, a, params), None

        return jax.lax.scan(body, s, actions)[0]

    s = jax.vmap(lambda v: init_state(vx=v))(jnp.linspace(0.0, 60.0, args.n_cars))
    t0 = time.perf_counter()
    jax.block_until_ready(run(s, actions))  # compile + first run
    print(f"device: {jax.devices()[0]}  compile+first run: {time.perf_counter() - t0:.2f}s")

    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        # block_until_ready: JAX dispatch is async, so time the finished result, not the launch.
        s = jax.block_until_ready(run(s, actions))
        times.append(time.perf_counter() - t0)
    best = min(times)
    sps = args.n_cars * args.n_steps / best
    print(
        f"{args.n_cars} cars x {args.n_steps} steps: best {best * 1e3:.1f} ms -> "
        f"{sps / 1e6:.2f} M car-steps/s ({params.n_substeps} substeps each), "
        f"{sps * params.dt / 3600:.1f} simulated hours per wall second"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
