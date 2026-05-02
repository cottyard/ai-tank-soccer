# Neural Tank Soccer Project

## Goal

This repository exists to make the browser-playable 1v1 tank soccer AI stronger. The playable model is `public/models/neural-best.json`; replace it only after a candidate beats the current accepted model in deterministic runtime gates.

The AI should improve in live browser matches: kickoff contests, loose-ball recovery, corner fights, defending after losing possession, finishing, and stamina management.

Agents should make training and implementation decisions autonomously when they serve this goal. Ask the user only for product-goal changes, data-loss risk, private-data exposure, external spending, or conflicting instructions.

## Current Accepted Model

Current accepted weights:

- File: `public/models/neural-best.json`
- Promoting commit: `9b70a00 Improve tank AI with mixed-start PPO baseline`
- Candidate source: `training-runs/neural-pg-mixed-group-self-s2026050208.json`
- Training mode: native Rust PPO, runtime action execution, mixed starts, `start-team-time` advantage baseline, self-play opponent.

Runtime gate results against the previous accepted model:

| Gate | Previous | Current |
| --- | ---: | ---: |
| Standard seeds `[19, 31, 43, 57, 71]`, `matches=4`, `frames=600` | `avgScore=317.991`, goals `11-1`, `avgWin=0.725`, `avgBp=0.242` | `avgScore=320.213`, goals `11-1`, `avgWin=0.725`, `avgBp=0.270` |
| Holdout seeds `[83, 97, 109, 127, 149]`, `matches=4`, `frames=600` | `avgScore=348.098`, goals `12-1`, `avgWin=0.750`, `avgBp=0.267` | `avgScore=353.840`, goals `11-0`, `avgWin=0.750`, `avgBp=0.289` |

Extra diagnostic seeds are useful for overfit checks but are not the hard promotion gate unless explicitly chosen for a run.

## Architecture

- Browser runtime: TypeScript + Vite.
- Game model and deterministic physics: `src/game`.
- Runtime AI wrapper: `src/ai/neuralStrategy.ts`.
- Policy network and PPO-style updates: `src/ai/policyNetwork.ts`.
- Sparse-reward self-play collection: `src/ai/policyGradientTraining.ts`.
- Runtime deterministic gates: `src/ai/policyGate.ts`.
- Training harness: `scripts/coach-neural.ts`.
- Focused PPO CLI: `scripts/train-policy-gradient.ts`.
- Fast native trainer and sampler: `trainer-rust`.
- Diagnostic runtime search probe: `scripts/hill-climb-runtime.ts`; use only to inspect runtime sensitivity, not as the main promotion path.

Policy shape must remain compatible with the browser unless a coordinated migration is done:

- inputs: `36`
- hidden layers: `64, 64`
- outputs: `9`
- action mapping: `src/ai/policyActions.ts`

## Training Direction

Prefer self-play reinforcement learning over hand-authored tactical labels. Older supervised replay, curriculum labels, tactical rollouts, and traditional-strategy opponents may stabilize training, but they should not become the main intelligence source.

Current sparse-reward PPO supports:

- open, outcome-curriculum, and mixed starts;
- runtime-like action execution in the Rust sampler;
- frozen opponent weights;
- traditional opponent mode for diagnostics/stabilization;
- PPO clipping through old action probability;
- advantage normalization with `global` or `start-team-time` baselines;
- trainable-only sample filtering when playing against frozen or traditional opponents.

Rewards should stay tied to outcomes: future goals for/against and final win/loss/draw. Start-state curriculum is acceptable; shaped tactical rewards should be separate, documented, and used sparingly.

## Promotion Rules

Training completion is not success. A candidate is successful only if it beats the current accepted model under deterministic runtime gates.

Promotion flow:

1. Train a candidate into `training-runs/...json`.
2. Evaluate current accepted weights and the candidate on the standard gate.
3. Evaluate promising candidates on the holdout gate.
4. Promote only if score improves while goals and win proxy do not show a meaningful regression.
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

Targeted AI verification:

```powershell
npx vitest run tests/policyNetwork.test.ts tests/policyGradientTraining.test.ts tests/coachNeural.test.ts tests/nativePolicyTrainer.test.ts
```

Native PPO candidate training, current best-known starting point:

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

Runtime gate comparison template:

```powershell
@'
import { readFileSync } from 'node:fs';
import { loadWeightsPayload } from './scripts/coach-neural';
import { evaluateRuntimePolicy } from './src/ai/policyGate';

const candidates = [
  ['best', 'public/models/neural-best.json'],
  ['candidate', 'training-runs/neural-pg-candidate.json']
] as const;

for (const [name, path] of candidates) {
  const weights = loadWeightsPayload(readFileSync(path, 'utf8'));
  let gf = 0, ga = 0, score = 0, win = 0, bp = 0;
  for (const seed of [19, 31, 43, 57, 71]) {
    const r = evaluateRuntimePolicy(weights, { seed, matches: 4, frames: 600 });
    gf += r.goalsFor; ga += r.goalsAgainst; score += r.score; win += r.winProxy; bp += r.ballProgress;
    console.log(`${name} ${seed}: goals=${r.goalsFor}-${r.goalsAgainst} score=${r.score.toFixed(3)} win=${r.winProxy.toFixed(3)} bp=${r.ballProgress.toFixed(3)}`);
  }
  console.log(`${name} TOTAL goals=${gf}-${ga} avgScore=${(score / 5).toFixed(3)} avgWin=${(win / 5).toFixed(3)} avgBp=${(bp / 5).toFixed(3)}`);
}
'@ | npx tsx -
```

When internet research is needed from this environment, use the local HTTP proxy `http://127.0.0.1:10808`.

## Next Work Plan

1. Build an automated promotion loop that trains, runs standard gate, runs holdout gate, writes a concise metrics summary, and only then updates `public/models/neural-best.json`.
2. Add a real league sampler for native PPO: current accepted model, recent snapshots, selected historical accepted models, and traditional strategy as a low-weight stabilizer.
3. Add a learned value baseline or actor-critic path to reduce sparse-reward variance beyond grouped return baselines.
4. Expand mixed starts while keeping rewards outcome-based: gate-style open starts, outcome-curriculum starts, own-goal defense, side-wall/corner fights, and loose-ball contests.
5. Track candidate history in a small committed summary file or generated ignored metrics file, so future sessions can compare hyperparameters without reading every `training-runs/` artifact.
6. Keep runtime-action Rust parity covered by tests whenever `neuralStrategy`, stamina regulation, tactical rollout, or physics changes.

Avoid spending mainline time on output-bias hill climbing, stamina-threshold tuning, or local black-box weight probes. They overfit quickly and should remain diagnostics only.

## Repository Hygiene

Commit:

- source files,
- tests,
- documentation,
- package manifests,
- accepted model weights in `public/models/neural-best.json`.

Do not commit:

- `node_modules/`,
- `dist/`,
- `training-runs/`,
- browser replay JSON files,
- logs,
- native build outputs,
- Rust target directories,
- local environment files,
- coverage output.

Browser replay files are valuable but may contain user-specific data. Keep `browser-replay-*.json` out of commits unless intentionally promoted into a documented benchmark.
