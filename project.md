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

- open, outcome-curriculum, own-goal-defense, corner-fight, loose-ball-contest, and mixed starts;
- runtime-like action execution in the Rust sampler;
- frozen opponent weights;
- traditional opponent mode for diagnostics/stabilization;
- weighted league opponent sampling from current, recent, historical, and traditional opponents;
- PPO clipping through old action probability;
- advantage normalization with `global`, `start-team-time`, or `learned` baselines;
- trainable-only sample filtering when playing against frozen or traditional opponents.

Rewards should stay tied to outcomes: future goals for/against and final win/loss/draw. Start-state curriculum is acceptable; shaped tactical rewards should be separate, documented, and used sparingly.

## Development Principles

At the end of every work session, evaluate each training method and existing mechanism against the latest gate results, diagnostics, and implementation evidence. Keep methods that produce measurable progress, keep uncertain methods only as documented diagnostics, and remove or de-prioritize mechanisms that repeatedly fail to improve the accepted runtime policy.

When local evidence does not show a clear next path for improving training, use internet research to look for stronger reinforcement learning, self-play, optimization, or evaluation methods. From this environment, route that research through the local HTTP proxy `http://127.0.0.1:10808`, prefer credible primary sources, and record only ideas that can be tested in this project.

Maintain `project.md` as the working memory for the project. Update it promptly when commands, accepted models, training results, useful lessons, or rejected approaches change; keep useful experience, remove stale or low-signal information, and avoid preserving failed experiments as recommendations.

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

Automated native PPO promotion loop:

```powershell
npx tsx scripts/promote-policy-gradient.ts
```

This trains a candidate from `public/models/neural-best.json`, samples native PPO opponents from a weighted league, uses the learned value baseline for sparse-return variance reduction, cycles mixed starts across open, outcome-curriculum, own-goal defense, corner fights, and loose-ball contests, runs the standard runtime gate and then the holdout runtime gate, writes `training-runs/neural-promotion-summary-s2026050208.json`, appends a compact entry to `training-runs/neural-promotion-history.jsonl`, and replaces `public/models/neural-best.json` only if both gates pass without meaningful goals or win-proxy regression. Use `--no-promote` for a dry run.

Recent 2026-05-02 rejected promotion attempts:

| Seed | Variant | Candidate Standard Gate | Decision |
| --- | --- | --- | --- |
| `2026050208` | `learned` baseline + weighted `league` opponent, `lr=0.001`, `epochs=2` | goals `9-1`, `avgScore=265.482`, `avgWin=0.700`, `avgBp=0.275` versus current goals `11-1`, `avgScore=320.213`, `avgWin=0.725`, `avgBp=0.270` | Reject; de-prioritize this exact recipe until modified. |
| `2026050209` | `start-team-time` baseline + `self` opponent, `lr=0.001`, `epochs=2` | goals `10-1`, `avgScore=290.814`, `avgWin=0.675`, `avgBp=0.268` | Reject; closer on goals but still lower score/win proxy. |
| `2026050210` | same as above with `lr=0.0005` | goals `8-1`, `avgScore=238.851`, `avgWin=0.675`, `avgBp=0.293` | Reject; reducing learning rate alone made gate performance worse. |
| `2026050211` | same as above with `epochs=1` | goals `8-1`, `avgScore=237.964`, `avgWin=0.675`, `avgBp=0.282` | Reject; reducing epochs alone made gate performance worse. |

Internet research via `http://127.0.0.1:10808` found two immediately testable ideas from primary sources: PPO's clipped surrogate is intended to support multiple minibatch epochs but remains hyperparameter-sensitive ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)); Population Based Training searches hyperparameters and schedules under a fixed budget instead of relying on one fixed recipe ([Jaderberg et al. 2017](https://arxiv.org/abs/1711.09846)). For this project, prefer a small promotion-oriented PBT/grid runner over more one-off full-length runs. MuZero-style learned-model planning ([Schrittwieser et al. 2019](https://arxiv.org/abs/1911.08265)) is too large a jump for the current codebase and should stay out of the near-term path.

League opponents can include the current accepted model, extra snapshots, and a low-weight traditional stabilizer:

```powershell
npx tsx scripts/promote-policy-gradient.ts `
  --league-opponent-weights training-runs/recent-snapshot.json `
  --league-current-weight 1 `
  --league-traditional-weight 0.15
```

Short promotion-oriented PPO grid search:

```powershell
npx tsx scripts/search-policy-gradient.ts `
  --seed 2026050212 `
  --matches 240 `
  --frames 180 `
  --gate-matches 2 `
  --gate-frames 360 `
  --learning-rates 0.001,0.0008,0.0006 `
  --epochs-list 1,2 `
  --ppo-clips 0.08,0.12,0.16 `
  --temperatures 1.0,1.1 `
  --advantage-baseline start-team-time `
  --opponent-mode self
```

This writes ranked short-run candidates under `training-runs/policy-gradient-search-s<seed>/`, writes a JSON summary, appends `training-runs/policy-gradient-search-history.jsonl`, and does not promote weights. Use the best survivor as input to the full promotion loop before replacing `public/models/neural-best.json`.

Evaluate a search survivor through the full promotion gates without retraining it:

```powershell
npx tsx scripts/promote-policy-gradient.ts `
  --candidate-input training-runs/policy-gradient-search-s2026050217/v01-lr0p001-e1-clip0p12-t1p1.json `
  --summary-output training-runs/neural-promotion-search-survivor-summary-s2026050217.json
```

Search notes: a too-short `gate-frames=240` search produced no goals and no ranking signal, so short searches should keep the full `gate-frames=600` when possible. The `2026050217` two-variant search (`matches=120`, `frames=120`, full standard gate) found `lr=0.001`, `epochs=1`, `ppoClip=0.12`, `temperature=1.1` with standard delta `+0.447`, goals unchanged at `11-1`, and holdout unchanged at goals `11-0`, `avgScore=353.840`, `avgWin=0.750`, `avgBp=0.289`. The full `--candidate-input` promotion check rejected it because the holdout score did not improve. Keep this as a non-regressing survivor and useful search signal, not an accepted promotion.

When using `--candidate-input`, the promotion loop reads the candidate weight metadata for seed, baseline, opponent mode, epochs, batch size, learning rate, clip, temperature, discount, start mode, and action mode. This keeps promotion summaries and history tied to the actual search candidate instead of the promotion loop's default training recipe.

When internet research is needed from this environment, use the local HTTP proxy `http://127.0.0.1:10808`.

## Next Work Plan

1. Expand search toward candidates that improve holdout, not just standard: run small full-gate searches over `ppo-clip`, `temperature`, and random seed while keeping `gate-frames=600`.
2. Keep learned baselines and weighted league sampling available, but treat them as variants to search rather than defaults.
3. Keep runtime-action Rust parity covered by tests whenever `neuralStrategy`, stamina regulation, tactical rollout, or physics changes.

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
