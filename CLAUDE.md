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
- Friction circle per axle on. Drag on, downforce behind `cla`, off. Generic GT3-ish car.
- Physics 60 Hz (4 substeps), policy 20 Hz (action repeat 3), gamma 0.99.
- Track limits: penalty when all four wheels are over the edge; terminate 3 m beyond that.
- Checkpoint selection on held-out procedural val tracks; reference tracks only in M5.
- Reward: 0.01/m progress, -0.05 per decision past limits, -0.01 * ||delta action||^2.
  Crash penalty (`crash_penalty`) exists but is 0: owner dropped it 2026-09-26 after the GPU A/B,
  and raised the jerk penalty 0.002 -> 0.01 against steering weave. Gamma stays 0.99.

## Gotchas found so far
- Physics substeps are separated by `jax.lax.optimization_barrier`. Without it XLA fusion blows up
  super-linearly (500x slower on CPU). Re-benchmark with and without it on GPU.
- Weakly-typed leaves in the train runner cause a recompile on the second chunk; create arrays
  with explicit dtypes.
- The car is part of the run config (`--car.*`, saved in config.json). Eval and its pure-pursuit
  baseline use the checkpoint's car. Runs saved before this field existed load the default car.
- Aim local repos need indexing (`aim up` or `aim storage reindex`) before the SDK can read runs.

## Status
- M1-M3 verified on CPU. M4 pipeline verified (smoke run, tests, Aim logging).
- CPU probe (512 envs, 30M steps, 10% of the planned budget): val clean-lap rate 0.17 and still
  rising when the LR schedule hit zero. M5 on it: sweeper clean (55.9 s vs pure pursuit 53.1 s),
  oval and hairpin crash. Failure mode is corner over-speed (median crash speed 1.7x the pure
  pursuit corner speed on val), not a reward exploit. Mild steering weave on the sweeper.
- Crash penalty added after that probe (see Decisions). Same-seed CPU probe with it: slower
  (133 vs 152 km/h mean), crashes at lower speed but about as often (53 vs 50 of 64 val tracks),
  best val clean 0.11 vs 0.17. M5: oval now clean, hairpin still crashes, sweeper 67 s vs 56 s and
  more steering weave. Single seed at 10% budget, so not conclusive; needs a GPU A/B.
- GPU box = this PC's RTX 2060 via Docker Desktop (`--gpus all`, venv in volume `trace-venv`,
  Aim in volume `trace-aim`); see `runs/queue.sh`. JAX can't preallocate 6 GB there, falls back
  to ~4 GB and runs fine. M1: 252 M car-steps/s at 4096 cars. Training ~860k sps, 300M in ~7 min.
- GPU A/B 2026-09-25, 300M, seeds 0-2, best val clean: crash penalty 0.25/0.44/0.44, none
  0.13/0.47/0.36. Inconclusive; no penalty drives ~12% faster and beats pure pursuit on the oval
  (43.0 vs 45.5 s). All 7 runs crash the hairpin (13.1 m radius, ~p5 of the train pool).
  Entropy collapses by ~50M steps (ent_coef 0). 1.5B-step run: val clean 0.48, sweeper 36% of
  time on the edge; nopen-s1 weaves at 14 steer reversals/s on the sweeper. No fix applied yet.
- Pending on the homelab (via the Arche MCP): Aim server and persistent checkpoint storage.
