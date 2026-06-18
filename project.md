# Tank Soccer Heuristic Learning Project

## Goal

This repository exists to make the browser-playable 1v1 tank soccer AI stronger.

The project direction is now **Heuristic Learning (HL)** instead of pure neural-network training. The playable AI should improve through an executable loop:

1. Run deterministic matches and traces.
2. Identify concrete failure modes from logs, replays, seed splits, or decision traces.
3. Convert the lesson into runtime policy code, tactical rollout scoring, tests, replay checks, or gate diagnostics.
4. Re-run standard and holdout gates.
5. Keep the change only if browser-runtime behavior improves without cross-seed safety regressions.

The neural weights in `public/models/neural-best.json` remain a useful base policy, but new mainline work should not assume that training new weights is the primary path to progress.

## Current Playable AI

The runtime policy is now best understood as:

- accepted neural base weights: `public/models/neural-best.json`
- runtime heuristic wrapper: `src/ai/neuralStrategy.ts`
- short-horizon tactical action search: `src/ai/tacticalRollout.ts`
- deterministic promotion and trace gates: `src/ai/policyGate.ts`

Current accepted weights:

- File: `public/models/neural-best.json`
- Promoting commit: `9b70a00 Improve tank AI with mixed-start PPO baseline`
- Candidate source: `training-runs/neural-pg-mixed-group-self-s2026050208.json`
- Training mode: native Rust PPO, runtime action execution, mixed starts, `start-team-time` advantage baseline, self-play opponent.

The weights are not replaced unless a candidate beats the accepted runtime gate. Runtime heuristic changes are accepted only when source and tests move together and full standard/holdout gates improve or remain safe.

## Heuristic Learning Research Route

Mainline research should use this loop:

- **Observe:** evaluate per-seed score, goals, win proxy, ball progress, action histograms, stamina, pressure signals, and first final-action divergences.
- **Diagnose:** choose one failure pattern, such as stuck corner balls, own-goal danger, low-stamina contact, loose-ball recovery, or missed finish setup.
- **Encode:** turn the lesson into executable behavior: tactical rollout trigger/scoring, a pressure signal, a guarded runtime rule, a regression state, a trace export, or a gate metric.
- **Verify:** run focused tests, then standard seeds `[19, 31, 43, 57, 71]` and holdout seeds `[83, 97, 109, 127, 149]` with `matches=4`, `frames=600`.
- **Record:** update this file with the hypothesis, code change, gate result, and next failure to study.

Pure PPO/neural training is now diagnostic only. It can still be used to generate hypotheses, candidate behaviors, or state anchors, but a training run is not considered progress unless it creates a behavior that survives the runtime wrapper and full gates.

## Bottleneck And Infrastructure Audits

Periodically audit whether the project is blocked by its current technical approach rather than by a missing local heuristic. This includes the browser/runtime split, TypeScript simulation speed, tactical rollout model, trace format, evaluation gates, training tools, test strategy, build system, native helpers, and any other infrastructure that affects learning velocity or AI quality.

The stack is not fixed. If a bottleneck is blocking measurable AI improvement, it is valid to change the architecture, rewrite core infrastructure, move more logic into native code, replace or extend the rollout/search system, add new analysis pipelines, change data formats, or otherwise migrate the technical foundation. Treat these changes as first-class project work when they are tied to a concrete bottleneck hypothesis, preserve or intentionally migrate deterministic browser-runtime behavior, include verification gates or migration tests, and are recorded in this file.

When progress stalls, also audit whether the neural policy, traditional strategy, runtime heuristics, tactical rollout, or any hybrid boundary is being used in the right place. Check for areas where a trained neural model could outperform hand-coded behavior, where traditional logic is still a better safety guard, and where the current split prevents learning from being tried. Training a focused neural component, replacing a traditional subsystem, or changing which policy controls a state class is valid project work when it is tied to a bottleneck hypothesis and verified through the same deterministic gates.

Do not keep a tool, language boundary, runtime architecture, or workflow merely because it is already in the repository. The standard is whether it helps produce a stronger playable AI safely and repeatably.

## Current HL Result

2026-06-18 HL iterations:

1. Attacking-corner tactical rollout is no longer skipped when the opponent is close to the ball.
2. Slow pinned attacking-corner balls now use a longer tactical rollout horizon.
3. Slow high-pressure attacking stalls can use a two-step tactical rollout sequence.
4. Slow central finish stalls can use a shorter two-step tactical rollout sequence.
5. Fast own-goal threats now use a longer defensive tactical rollout horizon.

Previous behavior: `shouldUseTacticalRollout` skipped rollout in attacking corners if an opponent was near the ball. In traced fragile seeds, this let the neural policy keep low-value single-track corner actions and left the ball pinned near the end wall or side wall. The lesson was converted into code by letting attacking-corner states use tactical rollout even under opponent pressure, and into tests by asserting that opponent-close attacking corners still run rollout and can override a bad raw action.

The second loop found another corner-specific failure in standard seed `31`: after the first fix, match `2` pushed the ball into a slow, pinned attacking corner, but the 18-frame rollout still preferred short-term safe actions. A targeted diagnostic showed that longer horizons can see the corner release payoff, so `tacticalRollout` now uses a 120-frame horizon only for slow pinned attacking-corner states. This is guarded by a regression test that verifies the default horizon finds a much better release than the short window.

The third loop found that some near-goal and side-wall stalls are not fixed by forcing a single forward action. Fixed-action probes from states such as `57:1`, `31:0/1`, and `19:3` often failed to score even over long horizons. The accepted lesson is narrower: when the ball is slow, attacking pressure is high, stamina is healthy, and the tank is close enough to influence the ball, tactical rollout scores a two-step sequence (first action, then best follow-up action) instead of a single repeated action. Low-stamina states stay on the old single-action rollout so critical stamina finish tests remain protected.

The fourth loop split central finish stalls from wall/corner stalls. Seed `57:1` had a slow ball in the goal lane with enough stamina and distance for a setup touch, but single-action rollout repeatedly preferred stop. Enumerating two-action sequences from the exact frame showed that a short setup action followed by a follow-up push can convert the chance. The accepted change adds a shorter sequence profile only for slow, high-stamina, central goal-mouth states; the existing longer two-step profile remains for wider attacking stalls.

The fifth loop targeted the only remaining standard-gate loss, seed `57:0`. The failure happened early: a fast ball in the own-goal lane was moving toward the red goal while the runtime rollout used an 18-frame window that preferred short-term reverse pressure. Longer diagnostic windows saw that this path carried own-goal risk, so fast own-goal threats now use a 72-frame defensive rollout horizon. A regression state captures the exact defensive danger pattern and verifies that the longer default horizon rejects the short-horizon reverse action.

Gate comparison using the same accepted weights:

| Gate | Before | After |
| --- | ---: | ---: |
| Standard `[19,31,43,57,71]`, `matches=4`, `frames=600` | goals `11-1`, `avgScore=320.213`, `avgWin=0.725`, `avgBp=0.270` | goals `14-0`, `avgScore=433.364`, `avgWin=0.800`, `avgBp=0.242` |
| Holdout `[83,97,109,127,149]`, `matches=4`, `frames=600` | goals `11-0`, `avgScore=353.840`, `avgWin=0.750`, `avgBp=0.289` | goals `19-0`, `avgScore=565.393`, `avgWin=0.875`, `avgBp=0.163` |

Per-seed gains from this HL change:

- Standard seed `57` improved from `1-1`, score `27.416` to `2-1`, score `161.270`.
- Holdout seed `109` improved from `1-0`, score `189.493` to `2-0`, score `328.737`.
- Holdout seed `127` improved from `4-0`, score `615.111` to `5-0`, score `749.371`.
- Holdout seed `149` improved from `2-0`, score `330.360` to `4-0`, score `596.633`.
- Holdout seed `83` improved from `3-0`, score `460.512` to `4-0`, score `591.803`.
- Holdout seed `97` improved from `1-0`, score `173.726` to `2-0`, score `306.385`.
- Standard seed `71` improved from `2-0`, score `318.535` to `3-0`, score `453.296`.
- Standard seed `19` improved from the post-corner `2-0`, score `329.908`, win `0.750` to `3-0`, score `473.083`, win `0.875`.
- Holdout seed `109` improved from the post-corner `2-0`, score `328.737` to `3-0`, score `455.772`.
- Holdout seed `127` improved from the post-corner `5-0`, score `749.371` to `6-0`, score `876.373`.
- Standard seed `57` improved again from the post-two-step `2-1`, score `161.100`, win `0.625` to `3-1`, score `294.054`, win `0.750`.
- Standard seed `57` improved once more after the own-goal defensive horizon change from `3-1`, score `294.054`, win `0.750` to `3-0`, score `450.030`, win `0.875`.
- Standard seed `31` remains below the session-start baseline (`2-0`, score `324.809` to `1-0`, score `195.901`); this is still the next standard-gate diagnosis target.

Rejected follow-up ideas from the same session:

- A broad near-goal finish-stall scoring bonus/stop penalty passed local tests but regressed the gate to standard goals `11-2`, `avgScore=287.503`, `avgWin=0.700`; do not reintroduce it without a narrower hypothesis.
- A sandboxed force-forward rule for low-speed near-goal balls produced only tiny score deltas and no standard win-proxy gain; it is not worth committing.
- A guarded "keep active policy action" rule for slow high-pressure finishes avoided one local stall but did not improve the gate: standard stayed goals `12-1`, `avgWin=0.725` and score slipped slightly to `346.993`; reverted.
- A medium-horizon rollout for slow attacking side-wall balls helped a local `19:3` trace frame but produced no aggregate gate change; reverted until a scoring or contact model explains why the longer action does not convert.
- A goal-mouth setup / end-wall near-miss scoring probe fixed the local `31:2` path in one broad variant and raised a narrower variant's standard goals/score to `14-1`, `avgScore=400.090`, but standard `avgWin` stayed `0.750`, seed `31` remained `1-0`, holdout slipped to goals `18-0`, `avgWin=0.850`, and runtime gates slowed down. Keep the lesson as diagnostic only: near-miss scoring needs a cleaner contact/sequence model before it is worth committing.
- A narrow "keep raw forward instead of rollout stop" gate for high-stamina central finish stalls reproduced the `57:1` local pattern but did not change the standard gate: goals `13-1`, `avgScore=375.610`, `avgWin=0.750`. Do not keep raw-forward guards unless they create actual match conversion.
- A narrow side-wall finish sequence probe for `31:2` targeted the exact frame where the ball was deep in the attacking corner, rolling into the side wall at speed `18.939`, and the tank still had `0.593` stamina. A `12+48` two-step profile was locally plausible and did not hurt holdout, but it produced no standard-gate improvement: standard stayed goals `14-1`, `avgScore=402.135`, `avgWin=0.775`; holdout stayed goals `19-0`, `avgScore=565.393`, `avgWin=0.875`. Revert this class of wall-edge sequence unless the rollout evaluator or continuation model proves an actual match conversion.

Diagnostic tooling added after the corner fixes:

- `scripts/diagnose-runtime-failures.ts` emits match-level HL summaries with outcome, final ball/tank state, action histograms, pressure averages, and tail decisions.
- `scripts/probe-runtime-macros.ts` runs fixed-action counterfactuals against the browser-runtime policy. It now caches aligned start states and forks from those frames, which keeps broad macro sweeps practical without changing AI-decision-frame semantics.
- Standard failure snapshot after the accepted corner changes: goals `12-1`, wins `10`, draws `9`, losses `1` over 20 standard matches. The two-step sequence improves this to goals `14-1`, wins `12`, draws `7`, losses `1`; the fast own-goal defensive horizon improves it again to goals `14-0`, wins `13`, draws `7`, losses `0`.
- Main observed failure patterns: high-pressure near-goal stalls where tactical rollout prefers stop over the raw forward action (`57:1`, `31:0/1`), and attacking side-wall/corner stalls where longer horizons can see local movement but still do not change match outcomes (`19:3`, `31:2`).

Decision: keep the five HL runtime changes if full tests/build pass. Standard win proxy improves from `0.725` at session start to `0.800` (+10.3%), while holdout win proxy improves from `0.750` to `0.875` (+16.7%) and holdout goals rise from `11-0` to `19-0`. The rough +10% standard objective is now met; continue with seed `31` as the main remaining under-baseline standard case.

2026-06-19 continuation from the accepted `avgWin=0.800` HL baseline:

The next target was another relative +10% from the current accepted baseline. Because the standard gate has 20 matches and win proxy moves in `0.025` increments, the practical target was `avgWin >= 0.900`.

Accepted additions:

1. Low-pressure contact now recovers stamina instead of spending weak contact energy when stamina is below `0.34`, finishing pressure is low, own-goal pressure is low, ball speed is modest, and the tank is already in contact range.
2. Safe own-corner releases can wait when the ball is rolling away from the own goal and there is no urgent goal threat.
3. Low-stamina drifting finish chances can wait when the ball is already in a high-quality finish lane and a rushed touch is more likely to spoil the setup.
4. Critical-stamina rolling finish pushes can preserve full forward drive only for a narrow pattern: high but not maximal finish pressure, no major own-goal risk, the ball in the goal lane, low lateral drift, controlled forward attack velocity, and the raw/tactical policy already choosing full forward. This converted standard `19:1` while avoiding the nearby `31:0/1` regressions that appeared under broader variants.

Rejected during this continuation:

- Broad opponent predicted-action assumptions inside rollout regressed standard performance.
- Wider low-pressure recovery thresholds (`0.38`/`0.42`) lost the accepted gains.
- Lane-progress position scoring, exhausted-finish stamina recovery, longer low-stamina finish rollout horizons, broad critical-stamina finish push preservation, and widened drifting-wait thresholds either produced no gate win or traded one converted draw for another regression.
- A broader rolling-finish push converted `19:1` but regressed seed `31`; adding the low-lateral-drift condition was required for the final accepted version.

Updated gate comparison from the previous accepted `0.800` baseline:

| Gate | Previous Accepted | Current |
| --- | ---: | ---: |
| Standard `[19,31,43,57,71]`, `matches=4`, `frames=600` | goals `14-0`, `avgScore=433.364`, `avgWin=0.800`, `avgBp=0.242` | goals `17-0`, `avgScore=517.609`, `avgWin=0.900`, `avgBp=0.226` |
| Holdout `[83,97,109,127,149]`, `matches=4`, `frames=600` | goals `19-0`, `avgScore=565.393`, `avgWin=0.875`, `avgBp=0.163` | goals `19-0`, `avgScore=567.751`, `avgWin=0.875`, `avgBp=0.192` |

Per-seed current standard gate:

- `19`: `4-0`, score `608.975`, win `1.000`, ball progress `0.300`.
- `31`: `3-0`, score `466.834`, win `0.875`, ball progress `0.281`.
- `43`: `3-0`, score `471.716`, win `0.875`, ball progress `0.342`.
- `57`: `4-0`, score `583.354`, win `1.000`, ball progress `-0.021`.
- `71`: `3-0`, score `457.165`, win `0.750`, ball progress `0.230`.

Per-seed current holdout gate:

- `83`: `4-0`, score `596.182`, win `1.000`, ball progress `0.140`.
- `97`: `2-0`, score `309.350`, win `0.750`, ball progress `0.070`.
- `109`: `3-0`, score `458.186`, win `0.750`, ball progress `0.243`.
- `127`: `6-0`, score `878.000`, win `1.000`, ball progress `0.287`.
- `149`: `4-0`, score `597.036`, win `0.875`, ball progress `0.221`.

Current remaining standard draws after the accepted continuation are `31:0`, `43:1`, `71:0`, and `71:3`. The practical next target is to convert one of these without losing the `31` safety recovered by the low-lateral-drift guard.

Decision: keep the continuation if full tests/build pass. The standard gate improved from `avgWin=0.800` to `avgWin=0.900`, which is a relative +12.5% improvement from the accepted baseline and satisfies the requested additional +10% objective while holdout remains safe.

## Architecture

- Browser runtime: TypeScript + Vite.
- Game model and deterministic physics: `src/game`.
- Runtime AI wrapper: `src/ai/neuralStrategy.ts`.
- Short-horizon action search: `src/ai/tacticalRollout.ts`.
- Position scoring for rollout: `src/ai/positionEvaluation.ts`.
- Accepted neural weights and loader: `public/models/neural-best.json`, `src/ai/bundledPolicy.ts`.
- Policy network and old PPO tooling: `src/ai/policyNetwork.ts`, `src/ai/policyGradientTraining.ts`, `scripts/train-policy-gradient.ts`, `trainer-rust`.
- Runtime deterministic gates and decision traces: `src/ai/policyGate.ts`, `scripts/trace-runtime-policy.ts`.

Policy shape must remain compatible with the browser unless a coordinated migration is done:

- inputs: `36`
- hidden layers: `64, 64`
- outputs: `9`
- action mapping: `src/ai/policyActions.ts`

## Promotion And Acceptance Rules

Training completion is not success. A change is successful only if it improves or safely preserves deterministic browser-runtime behavior.

Runtime heuristic acceptance:

1. The change must have a concrete failure-mode hypothesis.
2. It must add or update a regression test when the behavior is local enough to test.
3. It must be evaluated on standard and holdout gates.
4. Keep it only if aggregate score/goals improve or remain safe and no severe seed-level regression is left unexplained.

Neural weight promotion:

1. Train a candidate into `training-runs/...json`.
2. Evaluate current accepted weights and the candidate on the standard gate.
3. Evaluate promising candidates on the holdout gate.
4. Promote only if score improves while goals and win proxy do not meaningfully regress.
5. Replace `public/models/neural-best.json` only after promotion.
6. Run full verification.
7. Commit source, tests, docs, and accepted weights together.

Rejected candidates stay in `training-runs/` for analysis and are not committed.

## Useful Commands

Install and run:

```powershell
npm install
npm run dev
```

Full verification:

```powershell
npm test
npm run build
cargo build --release --manifest-path trainer-rust\Cargo.toml
```

Runtime gate comparison:

```powershell
@'
import { readFileSync } from 'node:fs';
import { loadWeightsPayload } from './scripts/coach-neural';
import { evaluateRuntimePolicy } from './src/ai/policyGate';

const weights = loadWeightsPayload(readFileSync('public/models/neural-best.json', 'utf8'));
const groups = [
  ['standard', [19, 31, 43, 57, 71]],
  ['holdout', [83, 97, 109, 127, 149]]
] as const;

for (const [name, seeds] of groups) {
  let gf = 0, ga = 0, score = 0, win = 0, bp = 0;
  console.log(`GROUP ${name}`);
  for (const seed of seeds) {
    const r = evaluateRuntimePolicy(weights, { seed, matches: 4, frames: 600 });
    gf += r.goalsFor; ga += r.goalsAgainst; score += r.score; win += r.winProxy; bp += r.ballProgress;
    console.log(`${seed}: goals=${r.goalsFor}-${r.goalsAgainst} score=${r.score.toFixed(3)} win=${r.winProxy.toFixed(3)} bp=${r.ballProgress.toFixed(3)}`);
  }
  console.log(`TOTAL goals=${gf}-${ga} avgScore=${(score / seeds.length).toFixed(3)} avgWin=${(win / seeds.length).toFixed(3)} avgBp=${(bp / seeds.length).toFixed(3)}`);
}
'@ | npx tsx -
```

Focused tests for the current HL runtime:

```powershell
npx vitest run tests/neuralStrategy.test.ts tests/policyGate.test.ts tests/tacticalRollout.test.ts tests/positionEvaluation.test.ts
```

Decision trace diagnostics:

```powershell
npx tsx scripts/trace-runtime-policy.ts `
  --current public/models/neural-best.json `
  --seeds 31 57 97 109 `
  --matches 4 `
  --frames 600 `
  --decision-analysis `
  --output training-runs/runtime-trace-hl-diagnostic.json
```

Match-level HL failure diagnostics:

```powershell
npx tsx scripts/diagnose-runtime-failures.ts `
  --seeds 19 31 43 57 71 `
  --matches 4 `
  --frames 600 `
  --tail-decisions 30 `
  --output training-runs/hl-standard-failures.json
```

Native PPO tooling remains available for diagnostics:

```powershell
npx tsx scripts/train-policy-gradient.ts `
  --native `
  --input public/models/neural-best.json `
  --output training-runs/neural-pg-candidate.json `
  --metrics-output training-runs/neural-pg-candidate-metrics.json `
  --seed 2026050208 `
  --matches 960 `
  --frames 240 `
  --epochs 2 `
  --batch-size 192 `
  --learning-rate 0.001 `
  --ppo-clip 0.12 `
  --temperature 1.1 `
  --discount 0.996 `
  --start-state-mode mixed `
  --advantage-baseline start-team-time `
  --action-mode runtime `
  --opponent-mode self
```

## Lessons From The Neural Plateau

The old pure neural/PPO route produced many useful diagnostics but stopped producing promotable runtime AI. Key retained lessons:

- Deterministic browser-runtime gates are the only promotion evidence that matches the playable AI.
- Sparse PPO updates often stayed behavior-invisible under argmax and runtime wrappers.
- Behavior-visible candidates often traded one seed gain for another seed regression.
- Open-start PPO found holdout gains, especially around seed `109`, but repeatedly regressed standard seeds.
- Anchor datasets, retention losses, early-forward weighting, reward-scalar sweeps, and wrapper-sample weighting were useful plumbing checks but did not create a promotable model.
- Output-bias hill climbing, stamina threshold tuning, and local black-box probes overfit quickly and should remain diagnostics only.

The practical lesson is not that neural networks are useless. The lesson is that this game currently improves faster when failures become executable runtime knowledge.

## Next Work Plan

1. Use `scripts/diagnose-runtime-failures.ts` and cached `scripts/probe-runtime-macros.ts` sweeps to study the remaining standard draws: `31:0`, `43:1`, `71:0`, and `71:3`.
2. Prioritize `31:0` and `43:1` as high-finish-pressure draws. Avoid broad critical-stamina or near-goal force rules; previous broader variants converted one draw while regressing another.
3. Treat `71:0` as a separate midfield/low-finish problem rather than a finish-stall problem.
4. For `71:3`, inspect whether the failure is side-wall geometry, critical-stamina regulation, or rollout scoring before changing thresholds.
5. When progress stalls, explicitly audit whether the neural policy, traditional strategy, heuristic wrapper, or tactical rollout owns the right state classes. A focused neural component or policy-boundary change is acceptable if it targets a concrete failure and survives standard/holdout gates.
6. If further improvements stall on search speed, prioritize a runtime-parity native evaluator or a faster replay/counterfactual harness. The TypeScript gate is still slow, but cached macro forking has reduced one immediate counterfactual bottleneck.
7. After each work session, remove stale project notes, record whether the AI improved, commit the relevant source/tests/docs, and push the branch.

## Repository Hygiene

Commit:

- source files,
- tests,
- documentation,
- package manifests,
- accepted model weights only when promoted.

Do not commit:

- `node_modules/`,
- `dist/`,
- `training-runs/`,
- browser replay JSON files,
- logs,
- native build outputs,
- Rust target directories,
- local environment files,
- coverage output,
- saved web pages or extracted article asset folders.

Browser replay files and saved pages can contain user-specific data. Keep them out of commits unless intentionally promoted into a documented benchmark.

When internet research is needed from this environment, use the local HTTP proxy `http://127.0.0.1:10808`.
