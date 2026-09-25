# Learning with trace

A reading guide for studying this codebase alongside the
[MLandAIfun](https://github.com/viik2k/MLandAIfun) study plan. Nothing here changes the project.
Answer the questions in your study journal; write predictions down before running anything.

## Part 1: code tour

1. `physics.py`: `step()` and `_substep()`. Which lines are acceleration, velocity, position? Why
   blend to a kinematic model below 5 m/s? Why the `optimization_barrier`?
2. `track.py`: why is uniform 2 m spacing the key design choice (module docstring)? Why are the
   reference tracks never used in training?
3. `env.py`: `observe()`, `transition()`, `step()`. What 50 numbers does the policy see? How is
   reward computed, and why does `step()` compute a reset state every step?
4. `pure_pursuit.py`: what privileged information does it use that the policy never gets?

## Part 2: experiments

Throwaway changes on a scratch branch, reverted afterwards.

- Set `offtrack_penalty` to 0 and rerun `uv run pytest tests/test_env.py -s`. Does pure pursuit's
  behaviour change? Why or why not?
- `uv run python scripts/bench_physics.py` with `--n-cars` 256, 4096, 16384. Where does
  throughput stop scaling?
- Change `grip_frac` in `PPConfig` from 0.8 to 0.95. Faster laps, or off-track events?

## Part 3: reading ppo.py (study modules 7 to 10)

| Piece | Study module | Questions |
|---|---|---|
| `Config` | 8, 10 | What is the effective horizon of `gamma` at 20 Hz? How many decisions per update? |
| `ActorCritic` | 6 | Why shrink the actor's last layer? Why is `log_std` a parameter, not an output? |
| `to_env_action` | 4 | What range does `tanh` give, and how do throttle and brake get to [0, 1]? |
| `gaussian_logp` | 4, 9 | Read the comment: why does the tanh correction cancel in the PPO ratio? |
| `gae` | 8, 10 | Walk the scan backwards by hand for 3 steps. What happens at `done`? Is a truncated episode treated differently from a crash here? |
| `loss_fn` | 10 | Find the ratio, the clip, the value loss. Why normalise advantages? |
| `make_train_chunk` | 7 | What is scanned over, what is vmapped over, and where is the jit boundary? |
| `make_evaluate` | 5 | Which tracks does validation use, and why not the reference tracks? |

Then run it: `uv run python -m trace_rl.ppo --smoke`, and `uv run pytest tests/test_ppo.py -v`.

## Part 4: M5

`scripts/eval.py` compares a checkpoint with pure pursuit on the reference tracks and writes a
Rerun replay. Watch the replay for driving that scores well but isn't good driving (circling,
cutting, weaving). CLAUDE.md's Status section records what the runs so far have shown.
