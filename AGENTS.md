# AGENTS.md — DSC-EPGG Week 1–2 Implementation

## Mission
Implement the Week 1–2 scope for DSC-EPGG on top of `marl-emecom` with strict scientific constraints.

## Scope (in)
- Stage 0: bootstrap + interface discovery.
- Stage 1: environment correctness + observation wrapper + tests.
- Stage 2: PPO + communication integration + smoke validation.
- Stage 3: session logging + regime identifiability audit.

## Scope (out)
- PLRNN training and all Week 5+ analyses.

## Scientific constraints (must not violate)
1. Do not leak rewards/welfare/true `f` into agent observations.
2. Keep information-set separation:
   - Set A (agent obs): noisy `f_hat`, endowment, lagged social features, messages.
   - Set B (learning): own reward for return/advantage only.
   - Set C (logging): full ground truth including intended/executed actions and flips.
3. Environment remains communication-agnostic; wrapper handles message features/dropout.
4. Preserve pinned legacy environment API expected by upstream codebase.

## Required implementation order
1. Environment fixes first (multi-step `step()`, Sticky-f, tremble, Box obs space, unclamped `f_hat`).
2. Wrapper integration second (history + EWMA + message marginals/dropout + tensor adapter).
3. PPO third (GAE trajectory buffer, clipped objective, value + entropy, joint action+message log-probs).
4. Logging/audit fourth.

## Communication fallback control
- Attempt up to 2 focused debug cycles for joint comm PPO.
- If unstable after 2 cycles, ship no-comm PPO baseline first.
- Re-enable comm in follow-up patch.

## Validation gates
### Gate 1 (before PPO)
- Environment and wrapper unit tests pass.
- Short smoke run passes without NaNs/crashes.

### Gate 2 (before merge)
- PPO losses finite.
- Entropy non-collapsed.
- Intended/executed action logging integrity verified.
- Stage 2 smoke run passes.

## Testing expectations
- Add/maintain tests for env dynamics, tremble rate, payoff correctness, unclamped observations, wrapper dims/EWMA/lag/dropout, and GAE golden case.
- Prefer deterministic seeds and reproducible smoke commands.

## Working style
- Keep commits focused by stage.
- Document any contract mismatches discovered in upstream code.
- Do not change scientific assumptions silently; if change is needed, explain in commit/PR notes.
