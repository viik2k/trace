# trace

Learned racing AI, phase 1: a GPU-parallel racing simulator and PPO training loop in JAX. One
control policy takes track geometry as input and should lap unseen procedural tracks cleanly.

## Setup

Linux or WSL2 (native Windows has no JAX GPU support).

```bash
uv sync --extra cuda   # GPU box
uv sync                # CPU only
uv run python -c "import jax; print(jax.devices())"
```

## Milestones and how to verify each

| Milestone | Command |
|---|---|
| M1 physics | `uv run pytest tests/test_physics.py -v -s` |
| M1 benchmark | `uv run python scripts/bench_physics.py --n-cars 4096` |
| M2 tracks | `uv run pytest tests/test_track.py -v -s` |
| M2 generate pools | `uv run python scripts/make_tracks.py` (writes `data/`, needed by M4/M5) |
| M2 view a track | `uv run python scripts/view_track.py --ref hairpin` then `uv run rerun track.rrd` |
| M3 environment | `uv run pytest tests/test_env.py -v -s` (pure pursuit laps all reference tracks) |
| M4 tests | `uv run pytest tests/test_ppo.py -v` (GAE, action squashing, smoke train + checkpoint) |
| M4 smoke run | `uv run python -m trace_rl.ppo --smoke` |
| M4 train | `uv run python -m trace_rl.ppo --aim-repo aim://<host>:53800` (see `--help`) |
| M5 eval | `uv run python scripts/eval.py --ckpt runs/<run>/best.eqx` then `uv run rerun eval.rrd` |

Lint: `uv run ruff check . && uv run ruff format --check .`

## Layout

```
src/trace_rl/
  physics.py       dynamic bicycle + simplified Pacejka, step(state, action, params)
  track.py         procedural generator, reference tracks, padded batching, IO
  env.py           observation, reward, termination, auto-reset, eval rollout
  pure_pursuit.py  baseline controller
  viz.py           Rerun logging
  ppo.py           PPO, single file: network, rollout, GAE, update, train loop, checkpoints
scripts/           bench_physics, make_tracks, view_track, eval
tests/
```

## Design notes

- **Physics**: 60 Hz, 4 internal substeps. GT3-ish car, 370 kW rear drive, no downforce. Friction
  circle per axle. Below 5 m/s total speed it blends to a kinematic bicycle model.
- **Tracks**: resampled at uniform 2 m arc spacing, so arc length maps to an index by division
  and lookahead is a gather. Padded to 2560 points. Half run clockwise.
- **Env**: policy acts at 20 Hz (3 physics steps per decision). Observation is car-frame only: no
  global position.
- **Reward**: progress in metres × 0.01, minus 0.05 per decision past track limits (all four wheels
  over the edge), minus 0.002 × ‖Δaction‖².
- **Termination**: 3 m of runoff past track limits, or stuck below 1 m/s for 2 s. Truncation at
  3000 decisions (150 s).
- **Training**: one jitted call runs `updates_per_chunk` PPO updates, with no host round-trips.
  Between calls the host logs to Aim, runs a jitted standing-start lap on 64 held-out procedural
  tracks, and keeps `best.eqx` by clean-lap rate, then lap time. Reference tracks are only used
  in M5.

## Experiment tracker

Training logs to any Aim repo passed as `--aim-repo`: a local path, or `aim://host:53800` for a
remote server. On the tracker host:

```bash
aim init --repo /srv/aim
aim server --repo /srv/aim --port 53800 &   # remote tracking endpoint
aim up --repo /srv/aim --host 0.0.0.0 --port 43800   # web UI
```
