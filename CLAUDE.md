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
- Reward: 0.01/m progress, -0.05 per decision past limits, -0.002 * ||delta action||^2,
  -2.0 on termination (runoff or stuck; stuck included so stopping can't dodge it). Owner chose a
  crash penalty over raising gamma to fix corner over-speed. Gamma stays 0.99.

## Gotchas found so far
- Physics substeps are separated by `jax.lax.optimization_barrier`. Without it XLA fusion blows up
  super-linearly (500x slower on CPU). Re-benchmark with and without it on GPU.
- Weakly-typed leaves in the train runner cause a recompile on the second chunk; create arrays
  with explicit dtypes.
- Aim local repos need indexing (`aim up` or `aim storage reindex`) before the SDK can read runs.

## Status
- M1-M3 verified on CPU. M4 pipeline verified (smoke run, tests, Aim logging).
- CPU probe (512 envs, 30M steps, 10% of the planned budget): val clean-lap rate 0.17 and still
  rising when the LR schedule hit zero. M5 on it: sweeper clean (55.9 s vs pure pursuit 53.1 s),
  oval and hairpin crash. Failure mode is corner over-speed (median crash speed 1.7x the pure
  pursuit corner speed on val), not a reward exploit. Mild steering weave on the sweeper.
- Crash penalty added after that probe (see Decisions). Watch for timid driving: slow but alive.
- Pending on the GPU box: M1 benchmark numbers, full training run, M5 on a trained policy.
- Pending on the homelab (via the Arche MCP): Aim server and persistent checkpoint storage.
