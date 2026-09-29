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
- Reward: 0.01/m progress, -0.2 per decision past limits, -0.01 * ||delta action||^2.
  Crash penalty (`crash_penalty`) exists but is 0: owner dropped it 2026-09-26 after the GPU A/B,
  and raised the jerk penalty 0.002 -> 0.01 against steering weave. Gamma stays 0.99.
- Owner 2026-09-26 before R5: off-track penalty 0.05 -> 0.2, ent_coef 0 -> 0.003.
- Track pools (owner 2026-09-26): 25% of generated tracks, train and val, get a spliced-in 180 deg
  hairpin (`hairpin_prob`); train also keeps 50% with a corner <14 m. R5 pools are in `data/tight14`.

## Gotchas found so far
- Physics substeps are separated by `jax.lax.optimization_barrier`. Without it XLA fusion blows up
  super-linearly (500x slower on CPU). Re-benchmark with and without it on GPU.
- Weakly-typed leaves in the train runner cause a recompile on the second chunk; create arrays
  with explicit dtypes.
- The car is part of the run config (`--car.*`, saved in config.json). Eval and its pure-pursuit
  baseline use the checkpoint's car. Runs saved before this field existed load the default car.
- Aim local repos need indexing (`aim up` or `aim storage reindex`) before the SDK can read runs.
- Under Docker/WSL the driver can hold the GPU at 1200 MHz ("Idle" clock event reason): ~630k sps
  instead of ~870k. Fix from an admin shell: `nvidia-smi -lgc 1800,2145` (resets on reboot).

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
  time on the edge; nopen-s1 weaves at 14 steer reversals/s on the sweeper.
- Overnight 2026-09-26 (new reward: jerk 0.01, no crash penalty). Scripts in `runs/round*.sh`.
  Best val clean per seed:
  - R1, 300M: ent_coef 0.003 wins with 0.38/0.34 (ent 0: 0.20/0.09; ent 0.01: 0.08/0.16). Sweeper
    weave gone (<1 reversal/s). 40% tight (<16 m) pool: same val, no hairpin gain at ent 0.003.
  - R2, 1.5B with ent 0.003: normal pool 0.86, tight pool 0.92. Laps all three reference tracks,
    but the hairpin goes off 4-16 times per 4 laps (edge cutting, since the off-track penalty is
    small). Ent 0.005 at 300M does no better than 0.003.
  - R3, 1.5B tight, seeds 1/2: 0.98/0.92. 3B: 0.94, so the budget has saturated.
  - R4, 1.5B, 50% <14 m pool (`data/tight14`), seeds 0-2: 0.86/0.91/0.97. s1 passes Phase 1:
    clean on all three reference tracks and beats pure pursuit (README). s0 crashes the hairpin;
    s2 laps it but goes off 10x on the oval. Passing on only 1 of 3 seeds is not robust yet.
- R5 2026-09-26 (`runs/round5.sh`): off-track 0.2, ent 0.003 and the 14 m pool as defaults, 1.5B,
  seeds 0-4. Val clean 1.0 on all 5, so val no longer separates runs. Full Phase 1 pass on s0 and
  s3 (2 of 5). s1 is clean everywhere but 0.85 s slower than pure pursuit on the hairpin. s2 and
  s4 still cut the hairpin (12-13 offs; the clean seeds are within ~2.6 s of their time). No crashes.
  Off-track 0.2 ended hairpin crashes but did not end cutting on every seed.
- 2026-09-26: the hairpin is out of distribution. It turns 182 deg within 60 m; no train or val track
  exceeds 122 deg (star-shaped control points cap it at ~120 even with more radius jitter or fewer
  points). A 64-track val set with min radius <14 m does not separate the R5 cutting seeds
  (s2 0.89, s4 0.97) from the clean ones (s0 0.92, s3 0.97).
- Hairpin pool (`track._splice_hairpin`): train 37% hairpins, val 25% (16 of 64). Pure pursuit laps
  all 16 val hairpins clean. R5 on them (clean tracks / off events): s0 9/12, s3 10/16, but the
  cutters s2 4/28 and s4 7/24, so the new val does separate them. R6 (`runs/round6.sh`, 5 seeds,
  1.5B) trains on it.
- R6 2026-09-27 (hairpin pool, R5 config otherwise): Phase 1 pass on 4 of 5 seeds (R5: 2 of 5).
  Hairpin clean on all 5 (0 offs, 56.3-57.3 s vs pure pursuit 61.5; R5 cutters were 56-58 s).
  s1 fails the sweeper instead: 5 offs, one per lap, all at s ~783 m, the inside of the 97 m right
  after a long gentle left, at 177 km/h (PP 130), 0.3-0.4 s over. Same in best and last. Val clean
  1.0, so the 64 val tracks miss it. s2 passes but rides the edge (16-29% of the time). No weave
  (rev/s <= 1.3).
- R7 2026-09-27 (`runs/round7.sh`, R6 config, seeds 5-9): clean on all three reference tracks on
  all 5, Phase 1 pass on s5, s6, s9. s7 ties pure pursuit on the sweeper (53.00 s) and s8 is
  slower (56.25 s): both cap their speed (s8 tops out ~208 km/h on the sweeper straights vs
  235-245 for s5 and PP), a conservative local optimum, not an exploit. s6/s9 ride the sweeper
  edge 36% of the time (legal). R6+R7: 7 of 10 seeds pass, 9 of 10 clean.
- 256 fresh tracks (seed 4242, hairpin_prob 0.25, 150 s from a standing start; scratch script): R6
  best.eqx goes off on 11-22 tracks and crashes on 0-5 (s1, the M5 failure, is cleanest at 11/4).
  R5-s2 (a hairpin cutter): 45/14. So 3 reference tracks and a saturated 64-track val are noisy
  robustness measures. last.eqx is no cleaner than best.eqx, so selection is not the issue.
  About half the failing tracks have a hairpin (20% of the pool); failing min radius p50 ~14 m vs
  18.5 m. R7 s5-s9: 9-18 tracks with offs, 1-6 crashes, so ~6% of fresh tracks have an off
  and ~1% a crash across the 10 seeds. Pure pursuit: 0/256 offs, 0 crashes. Off events are mixed: s0 40 inside / 10 outside,
  s1 10/13, s3 26/24 (R5-s2 38/67), so partly cutting, partly running wide.
- R8 2026-09-27 (`runs/round8.sh`): train pool hairpin_prob 0.5 (`data/hp50`, 60% hairpins), R6
  val, seeds 0-4. M5: all 5 clean, 4 pass (s2 speed-capped, sweeper 54.25 s); hairpin 55.3-56.9 s
  (R6 56.3-57.3). Fresh 256 vs R6, seeds 0-4 summed: tracks with offs 84 vs 81, off events 218
  vs 193, crashes 6 vs 16, and offs are now nearly all cuts (186 inside / 32 outside). So more
  hairpins fixed running wide and most crashes, not cutting; the lever left for cutting is the
  off-track penalty (owner's call). R9 (`runs/round9.sh`): hp50 seeds 5-9, for 10 vs 10.
  Hairpin-pool pass rate so far: 11 of 15 seeds, 14 of 15 clean on all reference tracks.
- R9 (hp50, seeds 5-9): pass on s5, s6, s9; s7 goes off 4x on the sweeper, s8 is clean but slower
  than pure pursuit there. hp50 is 7 of 10 seeds, the same as hp25 (R6+R7).
  `scripts/fresh_eval.py` is the fresh-256 check in the repo (it reproduces R8's 84 tracks with
  offs / 6 crashes). R9: 61/11. The hp50 control for ADR is 10 seeds: tracks with offs per seed
  21 24 20 11 8 13 10 11 11 16 (mean 14.5), crashes 17 in total.
- ADR (2026-09-29, `--adr`, default off; the default path is bit-identical on the smoke run):
  the train pool is sorted easy to hard (`track.by_difficulty`: rank of the tightest corner plus
  rank of max_turn). Resets draw from the first `cap` tracks. After each chunk the top band of
  the range (64 tracks = the val set's size, so the compiled evaluate is reused) runs like
  validation, and the cap moves one band on thresholds (0.5, 0.8). Track difficulty only: car
  randomisation is physics fidelity, the owner's call. Converged R6-s0 by band of its pool:
  clean 0.98 (easiest) ... 0.89 (hardest). So on hp50, ADR can only change the path to the
  full pool, not where it ends. A/B draft: `runs/round10.sh` (not run).
- Homelab 2026-09-26: Aim server on CT103 at `aim://192.168.4.103:53800` (UI on :43800). From
  round 7, `train()` in runs/lib.sh logs there and runs `sync_runs` after each run's eval (runs/
  goes to CT103, which PBS backs up nightly). If CT103 is down, training fails at Aim init: pass
  `--aim-repo /aim` to fall back to the local volume. End round scripts with `sync_runs` so
  roundN.txt gets copied too. See the README "Experiment tracker" section. Bump the server's aim
  together with uv.lock. Old `trace-aim` history (gpu-*, R1-R5) was migrated there 2026-09-26.
  R6 still logs to the local volume, and a follow-up `aim runs cp` pass copies it after round 6.
  Not done: a live reboot test of CT103.
  Onboarded 2026-09-26 (README "Homelab onboarding"). The UI is aim.arche.local (Authelia). Tracking
  is aim-track.arche.local:53800 (DNS to CT103, no auth). Wazuh FIM on /opt/arche/trace is proven,
  and the Kuma monitors are in Homelab `trace-monitors.json`. AIM= uses aim-track.arche.local, which
  passed the dk client test. If Pi-hole is down, fall back to 192.168.4.103.
