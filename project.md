# Tank Soccer AI

## Goal

Make the browser-playable 1v1 tank soccer AI stronger. A change counts as progress only when it improves deterministic browser-runtime behaviour and survives a measurement that can actually resolve it.

## Current Playable AI

- Neural base weights: `public/models/neural-best.json`
- Runtime heuristic wrapper: `src/ai/neuralStrategy.ts`
- Short-horizon action search: `src/ai/tacticalRollout.ts`
- Terminal valuation: hand-weighted `src/ai/positionEvaluation.ts` blended with the learned `src/ai/bundledValueModel.ts`
- Deterministic gates and traces: `src/ai/policyGate.ts`
- Large-sample benchmark: `src/ai/policyBenchmark.ts`, `scripts/benchmark-runtime.ts`
- Evolving paired opponent league: `src/ai/runtimeOpponentLeague.ts`, `config/runtime-opponent-league.json`

## How To Measure

Getting this wrong is how changes that were noise got accepted before.

The legacy standard `[19,31,43,57,71]` and holdout `[83,97,109,127,149]` gates run 4 matches per seed, so each is **20 matches total**. Their win proxy moves in `0.025` steps and its standard error is about `0.11`. They cannot resolve the size of effect this project actually produces. Treat them as reproducibility anchors only.

Use `scripts/benchmark-runtime.ts` for every strength claim:

- One physical start is played twice with sides swapped, and the scenario mean is the unit of observation, so side bias cancels.
- Seeds are drawn from a high range disjoint from gate and league seeds, so it measures generalisation instead of the seeds already tuned against.
- Identical strategies score exactly `0.5000` with zero variance. This is asserted in tests and is the harness's self-check.
- Against the accepted runtime the mirror control is exactly `0.5`, so a candidate is stronger only when its 95% interval excludes `0.5`.

Two rules that are not optional:

1. **Replicate on an independent `--salt` before believing anything.** A deeper rollout horizon scored `0.5212`, CI `[0.5026,0.5399]` over 200 scenarios and then `0.5095`, CI `[0.4978,0.5212]` over 500 fresh ones. Screening several variants and keeping the best one produces winner's curse; the legacy gate cannot detect it.
2. **A legacy gate movement is not evidence by itself.** Promoting the learned value model moved standard from `19-0`/`0.925` to `16-0`/`0.825` and holdout from `20-0`/`0.925` to `19-1`/`0.825`, which looks like a clear regression. Re-running the same seeds with 10x the matches showed the opposite on both:

| Gate seeds, 200 matches | Before | After |
| --- | ---: | ---: |
| standard | `0.7450 +-0.0604`, goals `124-6` | `0.7550 +-0.0596`, goals `122-5` |
| holdout | `0.7625 +-0.0590`, goals `126-4` | `0.7925 +-0.0562`, goals `134-2` |

The 20-match verdict was noise. Note also that the honest figure on those seeds is far below the `0.925` the 4-match gate advertises.

## What Is Known To Work

**A fast kernel.** `stepGame` recomputed `Math.cos`/`Math.sin` of the tank angle per polygon vertex inside all 16 collision iterations: `10658` trig calls per frame. Rewriting the collision path over reusable buffers with an angle cache and bounding-box rejection took it from `153.4us` to `6.0us` per frame (`25.6x`) with bit-identical trajectories. This is what makes large-sample measurement and value training affordable. At 5Hz decisions the browser now has roughly two orders of magnitude of headroom per decision, so runtime speed no longer limits search.

**Tactical rollout.** Strongly load-bearing, not overfitting. Against `traditional` over 800 matches it scores `0.7331 +-0.0190` against `0.4794 +-0.0128` with rollout disabled.

**A learned value function, blended.** `src/ai/valueNetwork.ts` regresses the signed, time-discounted identity of the next goal from the same 36 team-relative inputs the policy uses. Trained on 180000 Monte-Carlo samples it explains `64.8%` of holdout outcome variance. Blended at `0.08` into the rollout's terminal score it beats the previous accepted runtime `0.5189`, CI `[0.5096,0.5282]` over 1400 matches, replicated from `0.5288` on an independent sample, and improves play against `traditional` from `0.7512` to `0.7781`, paired delta `+0.0269`, CI `[0.0103,0.0434]`.

## What Is Measured Dead

Do not retry these without changing the underlying premise. Each was measured with confidence intervals, not anecdotes.

- **Deeper default rollout horizon.** No replicable gain; see the winner's curse example above.
- **Searching everywhere.** Bypassing `shouldUseTacticalRollout` scores `0.4738`, goals `62-83`. The trigger heuristic is protective. Midfield rollout was separately catastrophic at every improvement margin tried.
- **A reacting opponent inside rollout.** The search plans against a stationary opponent, because only the controlled tank gets a command and `sanitizeCommand` hands the opponent a full stop. Modelling the opponent as the same network re-deciding at 5Hz is a strictly better transition model and bought nothing: `0.5075`, CI `[0.4919,0.5231]`.
- **A purely learned terminal value.** Scores `0.3775`, goals `27-182`. Outcome prediction has almost no gradient over an 18-frame horizon, and the search actively seeks states where the model is wrong. The heuristic must supply dense local shaping; the learned model is only useful as a small blended correction.
- **Broad position-evaluation surgery.** Lowering the zero-speed `finishThreatScore` floor, narrowing the attacking-corner reward, and penalising deep non-scoring finish states each regressed the gate.
- **Direct action guards for individual stuck matches.** Near-goal finish-stall bonuses, force-forward rules, near-miss scoring, and follow-through guards produced local movement without match conversion.

The pattern across the first three is that search depth, coverage, and transition accuracy are all saturated. The binding constraint is terminal valuation.

## Architecture

- Browser runtime: TypeScript + Vite. Deterministic physics in `src/game`.
- Policy shape must stay browser-compatible: `36` inputs, hidden `64, 64`, `9` outputs, mapped in `src/ai/policyActions.ts`.
- Value network: `36` inputs, hidden `32, 32`, `1` output, embedded as a module rather than fetched so the browser, the gates, and the benchmark resolve identical weights synchronously.
- Native PPO tooling in `trainer-rust` remains available for diagnostics.

Deterministic trajectories are the entire evidence base, so `scripts/fingerprint-simulation.ts` folds every float of every frame into a digest and `tests/simulationFingerprint.test.ts` pins them. Physics digests must never change. A change that only affects decisions must leave all ten physics digests identical and move only runtime digests.

## Promotion Rules

1. State a concrete failure hypothesis.
2. Add or update a regression test when the behaviour is local enough to test.
3. Show a benchmark interval that excludes `0.5` against the accepted runtime, then replicate it on an independent `--salt`.
4. Check generalisation against a second opponent, and run the paired league.
5. Legacy gates are anchors: investigate any movement, but resolve it with more matches on the same seeds rather than accepting or rejecting on 20.
6. Re-baseline fingerprint digests in the same commit, and confirm physics digests are untouched.
7. Run full verification, then commit source, tests, docs, and weights together.

## Commands

```powershell
npm install; npm run dev
npm test; npm run build
```

Strength measurement:

```powershell
# Is a candidate stronger than the accepted runtime? Mirror control is exactly 0.5.
npx tsx scripts/benchmark-runtime.ts --policies accepted-runtime --opponent accepted-runtime --scenarios 400

# Replicate before believing it.
npx tsx scripts/benchmark-runtime.ts --policies <candidate> --scenarios 700 --salt 41

# Ablate. Variants take kind@key=value+key=value:
#   frames, margin, force, opp, value, blend
npx tsx scripts/benchmark-runtime.ts `
  --policies accepted-runtime,accepted-runtime@value=0,accepted-runtime@blend=0.15 `
  --opponent traditional --scenarios 400
```

Value model training:

```powershell
npx tsx scripts/train-value-network.ts --matches 900 --epochs 24 --output training-runs/value-network.json
npx tsx scripts/benchmark-runtime.ts --policies accepted-runtime@blend=0.08 `
  --opponent accepted-runtime --scenarios 700 --value-weights training-runs/value-network.json
```

Determinism and league:

```powershell
npx tsx scripts/fingerprint-simulation.ts
npm run gate:league:quick
npm run gate:league -- --details
npx tsx scripts/advance-runtime-league.ts   # preview; --write only after a promotion
```

Diagnostics: `scripts/diagnose-runtime-failures.ts`, `scripts/inspect-runtime-match.ts` (`--rollout-breakdown`, `--continuation-frames`), `scripts/probe-runtime-sequences.ts`, `scripts/trace-runtime-policy.ts`.

## Next Work

1. **Improve the value model, since it is the one lever that moved.** The blend is capped at `0.08` because a pure learned value collapses. Raising the usable blend weight is the direct path to a stronger AI. Likely routes: train on states drawn from rollout terminal distributions rather than on-policy play, so the model is accurate where the search actually queries it; predict a longer-horizon return; or add the heuristic's breakdown terms as inputs so the network only has to learn the correction.
2. **Retrain the value model against the promoted policy.** The current model was labelled by the pre-promotion runtime, so it is already one generation stale. Iterating this label-train-promote loop is the closest thing this project has to self-improvement.
3. **Ablate the accumulated heuristic constants on the benchmark.** Many were fitted to individual matches on the 20-match gate and cannot have been resolvable. Remove the ones that do not survive; expect several to be neutral.
4. **Do not spend effort on rollout depth, trigger coverage, or another guarded rule.** See what is measured dead.
5. Native or WASM porting is not justified for the playable AI while runtime headroom is ~100x. Revisit only if offline label generation becomes the throughput constraint, and only with cross-language parity added to the fingerprint suite.

## Repository Hygiene

Commit source, tests, documentation, package manifests, and accepted weights. Do not commit `node_modules/`, `dist/`, `training-runs/`, browser replay JSON, logs, native build outputs, Rust target directories, local environment files, coverage output, or scratch scripts. Replay files can contain user-specific data.

When internet research is needed, use the local HTTP proxy `http://127.0.0.1:10808`.
