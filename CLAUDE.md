# trace: notes for Claude sessions

Phase 1 of a learned racing AI: JAX sim + PPO, one policy that laps unseen procedural tracks.
Nothing beyond phase 1 gets built; later ideas go in DEFERRED.md with one line of reasoning.

## Working rules (from the owner)
- Minimal surface area. No speculative abstractions, config systems or plugin patterns.
- Every milestone is verified by a command that was actually run, with output shown (README).
- Ask before decisions with real cost: physics fidelity, reward design, compute. Just decide
  naming and minor style.
- Never silently tweak the reward. After each training run, review replays for exploitation
  (circling, cutting, oscillating) and propose fixes.
- Brief comments on non-obvious JAX decisions (jit boundaries, PRNG keys, pytrees).
- Stack: uv, Python 3.12, JAX, Equinox, Optax, tyro, Rerun, Aim, ruff, pytest.

## Decisions already taken
- Friction circle per axle on. Drag on, downforce deferred. Generic GT3-ish car.
- Physics 60 Hz (4 substeps), policy 20 Hz (action repeat 3), gamma 0.99.
- Track limits: penalty when all four wheels are over the edge; terminate 3 m beyond that.
- Checkpoint selection on held-out procedural val tracks; reference tracks only in M5.
- Reward: 0.01/m progress, -0.05 per decision past limits, -0.002 * ||delta action||^2.

## Gotchas found so far
- Physics substeps are separated by `jax.lax.optimization_barrier`. Without it XLA fusion blows up
  super-linearly (500x slower on CPU). Re-benchmark with and without it on GPU.
- Weakly-typed leaves in the train runner cause a recompile on the second chunk; create arrays
  with explicit dtypes.
- Aim local repos need indexing (`aim up` or `aim storage reindex`) before the SDK can read runs.

## Status
- M1-M3 verified on CPU. M4 pipeline verified (smoke run, tests, Aim logging); a CPU probe run
  shows learning (speed and return rising) but no GPU-scale run yet. M5 eval script verified on an
  untrained checkpoint.
- Pending on the GPU box: M1 benchmark numbers, full training run, M5 on a trained policy.
- Pending on the homelab (via the Arche MCP): Aim server and persistent checkpoint storage.
