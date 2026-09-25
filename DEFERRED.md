# Deferred

Out of scope for phase 1. One line of reasoning each.

## Later phases (from the brief)
- **Racecraft layer** (multi-car, overtaking, defending): needs a policy that can drive alone first.
- **Strategy layer** (tyres, fuel, pits): needs wear and fuel models that don't exist yet.
- **Behaviour cloning from LMU telemetry**: needs a telemetry pipeline and a mapping from LMU cars onto this physics model.
- **ONNX export to a Rust runtime**: only worth doing once a policy works; fixed observation scaling keeps the export stateless.

## Physics fidelity
- **Aero downforce on by default**: implemented behind `cla`, off; pure pursuit handles it, but it changes lap times and handling and should be evaluated on its own.
- **Load transfer on by default**: implemented behind `load_transfer`, off; turning it on shifts handling balance and should be evaluated on its own.
- **Lateral load transfer / four-wheel model**: the bicycle model has none; matters for kerbs and tyre temperature, not for clean laps.
- **Actuator dynamics** (steering rate limit, pedal lag): the jerk penalty stands in; add if the policy exploits instant inputs.
- **Tyre temperature and wear, surfaces, kerbs, grass grip**: not needed to lap cleanly.
- **Driver aids** (traction control, ABS): the policy should learn throttle modulation; only the pure-pursuit baseline has a simple traction budget.
- **Reverse gear**: stuck cars terminate instead.

## Learning
- **Car-frame centreline points in the observation**: fallback if curvature-only lookahead stalls learning (still no global position).
- **Randomised car parameters**: `CarParams` already vmaps over a batch of cars, but phase 1 trains one car.
- **Observation normalisation with running statistics**: fixed hand scaling is simpler and exports without state.
