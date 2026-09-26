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
| M4 train | `uv run python -m trace_rl.ppo --aim-repo aim://<host>:53800` (see `--help`; car variants via `--car.cla 3` etc.) |
| M5 eval | `uv run python scripts/eval.py --ckpt runs/<run>/best.eqx` then `uv run rerun eval.rrd` |

Lint: `uv run ruff check . && uv run ruff format --check .`

### M5 result (2026-09-26, RTX 2060 SUPER, ~35 min)

```bash
uv run python scripts/make_tracks.py --out data/tight14 --hairpin-prob 0
uv run python -m trace_rl.ppo --seed 1 --ent-coef 0.003 --total-steps 1500000000 \
  --train-tracks data/tight14/tracks_train.npz
uv run python scripts/eval.py --ckpt runs/<run>/best.eqx
```

```
track    driver       laps   flying offtrk  term  rev/s |dsteer|  edge%
oval     policy          5   42.20s      0 False   0.38    0.006   0.0%
oval     pure_pursuit    5   45.50s      0 False   0.00    0.001   0.0%
hairpin  policy          4   57.40s      0 False   1.74    0.016   4.2%
hairpin  pure_pursuit    3   61.50s      0 False   0.10    0.005   0.0%
sweeper  policy          4   51.00s      0 False   0.09    0.006   1.1%
sweeper  pure_pursuit    4   53.00s      0 False   0.00    0.001   0.0%
Phase 1: clean laps on all held-out tracks: True; beats pure pursuit on every flying lap: True
```

Val clean-lap rate 0.91. One seed of three passes; the other two went off or crashed on one track.

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

- **Physics**: 60 Hz, 4 internal substeps. GT3-ish car, 370 kW rear drive, downforce off
  (`cla`). Friction circle per axle. Below 5 m/s total speed it blends to a kinematic bicycle model.
- **Tracks**: resampled at uniform 2 m arc spacing, so arc length maps to an index by division
  and lookahead is a gather. Padded to 2560 points. Half run clockwise. Half the train pool has
  a corner under 14 m, and a quarter of generated tracks (train and val) get a spliced-in 180 deg
  hairpin, since the star-shaped generator alone never turns more than ~120 deg in 60 m.
- **Env**: policy acts at 20 Hz (3 physics steps per decision). Observation is car-frame only: no
  global position.
- **Reward**: progress in metres × 0.01, minus 0.2 per decision past track limits (all four wheels
  over the edge), minus 0.01 × ‖Δaction‖². A termination penalty (`--env.crash-penalty`) exists but
  defaults to 0.
- **Termination**: 3 m of runoff past track limits, or stuck below 1 m/s for 2 s. Truncation at
  3000 decisions (150 s).
- **Training**: one jitted call runs `updates_per_chunk` PPO updates, with no host round-trips.
  Between calls the host logs to Aim, runs a jitted standing-start lap on 64 held-out procedural
  tracks, and keeps `best.eqx` by clean-lap rate, then lap time. Reference tracks are only used
  in M5.

## Experiment tracker

Training logs to any Aim repo passed as `--aim-repo`: a local path, or `aim://host:53800` for a
remote server. Aim installs only on Linux (aimrocks has no Windows wheels), so on Windows the
client runs in the training container. The remote protocol is version-locked: the server's aim
must equal the client's in `uv.lock` (3.29.1).

The homelab server runs on CT103 (alduin-media, 192.168.4.103) as the compose stack
`/opt/arche/trace`. It has two services, `aim-server` (tracking, port 53800) and `aim-ui`
(http://192.168.4.103:43800). Both use a `python:3.12-slim` image with `aim==3.29.1` and
`restart: unless-stopped`. It is LAN only, with no auth. The Aim repo is in `/opt/arche/trace/aim`,
on the CT rootfs, so the nightly PBS job backs it up. A bare `aim server ... &` does not survive a
logout or a reboot, so use compose. Setup as run on 2026-09-26:

```bash
# on CT103, in /opt/arche/trace (Dockerfile + docker-compose.yml)
docker compose build
docker compose run --rm --no-deps aim-server aim init --repo /aim   # Initialized a new Aim repository at /aim
docker compose up -d
# aim-server  Up  0.0.0.0:53800->53800/tcp     aim-ui  Up  0.0.0.0:43800->43800/tcp
```

Smoke run from the Windows PC through the CUDA container (`dk` in `runs/lib.sh`):

```bash
source runs/lib.sh
dk "python -m trace_rl.ppo --smoke --aim-repo aim://192.168.4.103:53800"
# run runs/20260926-130927  devices [CudaDevice(id=0)]  2 chunks x 5 updates x 2048 decisions
# [2/2]     0.0M steps     155k sps | return  -0.64 ...
ssh root@192.168.4.103 docker exec aim-server aim runs --repo /aim ls
# e7881cdf590648ea84976e8e
# Total 1 runs.
```

That run has all 13 train metrics at 10 steps each and 2 val points on the server.

Checkpoints and evals (`runs/`) are copied to `/opt/arche/trace/runs` on the same CT, which is
also covered by PBS. The copy is additive: it never deletes on the server. It re-copies everything
each time, which is fine while runs/ is ~130 MB. `train()` in `runs/lib.sh` logs to this server
and runs `sync_runs` after each eval.

```bash
source runs/lib.sh && sync_runs
# 129M	/opt/arche/trace/runs
# 92
# real 0m1.872s
```

### Homelab onboarding (2026-09-26)

The stack is now codified in the Homelab repo (`media-stack/52-54`, `trace/`), the vault
As-Built note and Kuma (`lead-stack/uptime-kuma/trace-monitors.json`).

- UI: `https://aim.arche.local` through Traefik on CT100 with the local CA cert, behind Authelia.
- Tracking: `aim://aim-track.arche.local:53800`. This name is DNS only, pointing straight at
  192.168.4.103 and not through Traefik. The Aim client can't do Authelia's login redirect, and
  Traefik has no TCP entrypoint for it. So the tracking port stays LAN-only with **no auth**, as
  before.
- Wazuh: CT103's agent watches `/opt/arche/trace`'s own files (Dockerfile, compose) in realtime
  with diffs. `aim/` and `runs/` are excluded as data. Aim's container logs are not collected,
  because it logs almost nothing and no rule matches it. Proof that the watch works, from the
  manager's `alerts.json`:
  `2026-09-26T13:36:57Z alduin-media rule 554 lvl 5 File added /opt/arche/trace/.wazuh-fim-probe`,
  `13:37:02Z rule 553 lvl 7 File deleted` (the probe file was created and removed by hand).

Docker Desktop containers resolve `*.arche.local` through the host's resolver, which is Pi-hole.
The remote client works from the `dk` image on CPU. The venv was mounted read-only, so this
doesn't touch the GPU or a running round:

```bash
docker run --rm -v trace-venv:/venv:ro ghcr.io/astral-sh/uv:python3.12-bookworm sh -c \
  'getent hosts relnotes.arche.local; /venv/bin/python -c "from aim import Run
r = Run(repo=\"aim://192.168.4.103:53800\", experiment=\"homelab-dns-check\"); print(r.hash)"'
# 192.168.4.100   relnotes.arche.local
# d334f45f10594997b260a0f4
```

After the CT100 wiring (`bash media-stack/54-wire-core-trace.sh` in Homelab), the name works,
so `AIM=` in `runs/lib.sh` now points at it:

```bash
bash media-stack/54-wire-core-trace.sh
# dns aim: 192.168.4.100
# dns aim-track: 192.168.4.103
# https: 302 -> https://auth.arche.local/?rd=https%3A%2F%2Faim.arche.local%2F&rm=GET
curl --cacert ca.crt -o /dev/null -w '%{http_code} -> %{redirect_url}\n' https://aim.arche.local/
# 302 -> https://auth.arche.local/?rd=https%3A%2F%2Faim.arche.local%2F&rm=GET
#   (cert: issuer O=Arche Cluster, CN=Arche Local CA; subject CN=*.arche.local)
docker run ... (as above, repo="aim://aim-track.arche.local:53800")
# 192.168.4.103   aim-track.arche.local
# ok run 725b7cac66b543c1ada6a70f aim-track.arche.local:53800/.aim
```

`ca.crt` is `/opt/arche/core-stack/certs/ca.crt` on CT100. Windows curl (schannel) rejects the
cert without it until the CA is in the Windows trust store. If the name ever stops resolving
(for example, Pi-hole is down), set `AIM=aim://192.168.4.103:53800`.
