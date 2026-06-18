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

## Current HL Result

2026-06-18 HL iteration: attacking-corner tactical rollout is no longer skipped when the opponent is close to the ball.

Previous behavior: `shouldUseTacticalRollout` skipped rollout in attacking corners if an opponent was near the ball. In traced fragile seeds, this let the neural policy keep low-value single-track corner actions and left the ball pinned near the end wall or side wall. The lesson was converted into code by letting attacking-corner states use tactical rollout even under opponent pressure, and into tests by asserting that opponent-close attacking corners still run rollout and can override a bad raw action.

Gate comparison using the same accepted weights:

| Gate | Before | After |
| --- | ---: | ---: |
| Standard `[19,31,43,57,71]`, `matches=4`, `frames=600` | goals `11-1`, `avgScore=320.213`, `avgWin=0.725`, `avgBp=0.270` | goals `12-1`, `avgScore=346.449`, `avgWin=0.725`, `avgBp=0.260` |
| Holdout `[83,97,109,127,149]`, `matches=4`, `frames=600` | goals `11-0`, `avgScore=353.840`, `avgWin=0.750`, `avgBp=0.289` | goals `14-0`, `avgScore=436.084`, `avgWin=0.825`, `avgBp=0.262` |

Per-seed gains from this HL change:

- Standard seed `57` improved from `1-1`, score `27.416` to `2-1`, score `161.270`.
- Holdout seed `109` improved from `1-0`, score `189.493` to `2-0`, score `328.737`.
- Holdout seed `127` improved from `4-0`, score `615.111` to `5-0`, score `749.371`.
- Holdout seed `149` improved from `2-0`, score `330.360` to `3-0`, score `469.306`.
- Standard seed `31` regressed from `2-0`, score `324.809` to `1-0`, score `196.503`; this is now the next failure pattern to inspect before broadening the rule further.

Decision: keep the change if full tests/build pass, because aggregate standard and holdout gates improve substantially and goals do not regress overall. The standard seed `31` regression must be treated as the next HL diagnosis target.

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

1. Inspect the new standard seed `31` regression from the attacking-corner rollout change. Compare match-level ball paths and first final-action divergences against the previous behavior, then decide whether the rollout trigger needs a narrower guard or position scoring needs a corner-specific term.
2. Add a reusable HL diagnostic script or JSON summary format that records hypothesis, seeds, per-match goals, final ball/tank states, action histograms, and first divergences for one failure pattern.
3. Study late-match attacking-corner stalls where the ball is near the end wall and side wall with low velocity. Candidate mechanisms: longer rollout only in pinned attacking corners, a corner-release score term, or a macro target that moves the ball inward before shooting.
4. Keep neural training available only to generate candidate behaviors or state distributions. Do not start another PPO search until a specific runtime failure pattern demands it.
5. After each work session, remove stale project notes, record whether the AI improved, commit the relevant source/tests/docs, and push the branch.

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
