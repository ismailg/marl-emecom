# DSC-EPGG Week 1–2 Cloud Execution Plan

This repository is prepared for Codex Cloud delegation.

## Delivery stages

### Stage 0 — Bootstrap + Interface Discovery
1. Clone and install dependencies.
2. Lock environment (`requirements_locked.txt`).
3. Run baseline single-step script once.
4. Document actual contracts:
   - env observation type/shape
   - `step()`/`infos` structure
   - comm policy and auxiliary-loss hooks

### Stage 1 — Environment + Wrapper (Gate 1)
- Patch env for multi-step correctness and Sticky-f dynamics.
- Add trembling-hand execution noise before payoff.
- Remove `f_hat` clamping.
- Update observation space to `Box(shape=(2,), dtype=float32)`.
- Disable scalar-only assumptions (`gmm_ = False`, `normalize_obs = False`).
- Implement trainer-side `ObservationWrapper` with tensor adapter and message dropout.
- Pass Gate 1 tests and smoke run.

### Stage 2 — PPO + Communication (Gate 2)
- Implement trajectory buffer with intended/executed split and GAE.
- Implement PPO trainer with clipped surrogate, value, entropy.
- Jointly optimize action and message log-probs for sender agents.
- Reconnect signaling/listening auxiliary losses with configurable weights.
- End-to-end `train_ppo.py` rollout/update loop.
- Risk control: if joint comm PPO unstable after 2 debug cycles, ship no-comm PPO baseline first.

### Stage 3 — Logging + Regime Audit
- Session logger: `.npz` per session (Set C fields), plus consolidation.
- Regime audit: post-switch Bayesian convergence summary (`mean`, `median`, `p90`) and recommendation when `median <= 2`.

## Non-negotiable constraints
- No reward/welfare leakage into Set A observations.
- Env remains comm-agnostic.
- Preserve old 4-tuple env API if required by pinned upstream.

## Done criteria
- Gate 1 and Gate 2 pass.
- Tests cover env/wrapper/GAE/integration smoke.
- Output includes reproducible commands and a clean PR summary.
