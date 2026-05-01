# Neural Tank Soccer Project

## Project Goal

This project is a browser-playable 1v1 tank soccer game with an AI opponent. The long-term goal is to make the neural AI improve through self-play instead of hand-authored tactical patches.

The current practical objective is:

1. Keep the browser game playable and stable.
2. Keep the best accepted model in `public/models/neural-best.json`.
3. Build training pipelines that can generate candidate neural weights.
4. Accept a candidate only when it passes deterministic evaluation gates.
5. Gradually reduce dependence on traditional strategy and hand-written tactical labels as the neural model becomes stronger.

The user's observed target is simple and concrete: the AI should become a strong opponent in live browser matches, including kickoff contests, loose-ball recovery, corner fights, defense after losing possession, and stamina management.

Project-level autonomy:

- The only overriding purpose of this repository is to train a stronger tank AI.
- The user does not need to approve or confirm technical details, algorithm choices, implementation tactics, or training pipeline decisions.
- The agent should make those decisions autonomously when they serve the goal of stronger AI, then implement, verify, and report the outcome.
- Ask the user only when a decision changes the product goal, risks data loss, exposes private data, requires external spending, or conflicts with an explicit user instruction.
- This autonomy does not relax quality gates: accepted playable weights still require deterministic gate improvement before replacing `public/models/neural-best.json`.

## Current System Shape

The browser runtime is TypeScript and Vite. The game simulation is in `src/game`, and AI code is in `src/ai`.

Important modules:

- `src/game/model.ts`: game state, field constants, tank and ball model.
- `src/game/simulation.ts`: deterministic physics step.
- `src/game/match.ts`: fixed-timestep match simulation and AI clock.
- `src/ai/neuralStrategy.ts`: runtime neural policy wrapper used by the browser.
- `src/ai/policyNetwork.ts`: compact policy network, supervised training, and policy-gradient/PPO-style updates.
- `src/ai/policyActions.ts`: nine discrete tank-track actions.
- `src/ai/selfPlayTraining.ts`: older rollout-label self-play pipeline using tactical evaluation.
- `src/ai/policyGradientTraining.ts`: sparse-reward self-play RL pipeline.
- `src/ai/policyGate.ts`: deterministic runtime gate for candidate policies.
- `scripts/coach-neural.ts`: training harness with replay, curriculum, self-play, RL cycles, and acceptance gates.
- `scripts/train-policy-gradient.ts`: focused sparse-reward policy-gradient/PPO trainer.
- `trainer-rust`: Rust parity trainer for faster offline supervised policy training.
- `trainer-cpp`: earlier C++ trainer.

The deployed/playable neural weights are loaded from:

- `public/models/neural-best.json`

Do not overwrite this file just because training completed. A candidate must pass gates first.

## Research Direction

The preferred research direction is true self-play reinforcement learning.

Earlier training relied heavily on:

- Human replay behavior cloning.
- Generated curriculum labels.
- Short tactical rollouts.
- Hand-authored position evaluation.
- Traditional strategy as a teacher or gate opponent.

Those tools are still useful as stabilizers, but they should not become the main source of intelligence. The desired direction is closer to AlphaZero/PPO principles:

- The game rules and physics are known.
- The policy samples actions from its own network.
- Episodes produce rewards from actual game outcomes.
- Better behavior is reinforced through returns/advantages.
- Frozen historical opponents and league gates prevent overfitting to one self version.
- Search or curriculum may help exploration, but rewards should remain tied to actual outcomes.

Important distinction:

- It is acceptable to use start-state curriculum, such as near-goal, defensive, corner, and loose-ball starts.
- It is less desirable to use shaped tactical rewards that directly encode "go here", "turn this way", or "contest this way".
- If shaped rewards are added, document why they are necessary and keep them separate from sparse outcome rewards.

## Current RL Approach

The current sparse-reward pipeline is in `src/ai/policyGradientTraining.ts`.

It collects self-play decisions, records:

- policy inputs,
- sampled action,
- sampled action probability,
- log probability,
- team,
- frame,
- sparse future return,
- advantage,
- whether that decision belongs to the trainable current policy.

Rewards currently come from:

- future goals for/against,
- final win/loss/draw signal.

The pipeline supports:

- open self-play starts,
- outcome-curriculum starts,
- frozen opponent weights,
- PPO-style clipping through `oldProbability`,
- advantage normalization.

When using frozen opponent weights, only the current policy side should be trained. Opponent decisions may be logged for diagnostics, but must not enter the trainable sample set unless both sides intentionally share the same policy.

## Gate And League Rules

Candidate model adoption must be conservative.

The core rule:

> Training completion is not success. A candidate is successful only if it beats the current accepted model under the chosen gate.

Recommended acceptance flow:

1. Train a candidate into `training-runs/...json`.
2. Evaluate candidate against deterministic gate seeds.
3. Compare against the current `public/models/neural-best.json`.
4. Accept only if the candidate improves the gate score.
5. If accepted, replace `public/models/neural-best.json`.
6. If rejected, keep the candidate artifact for analysis but do not ship it.

Existing gate concepts:

- `traditional`: candidate must improve against the traditional strategy.
- `neural-default`: candidate must remain sane against the default neural weights.
- `neural-current`: candidate is checked against a frozen copy of the current accepted model.
- `league`: weighted combination of traditional, current neural, and default neural opponents.

League gating is preferred once the neural model is strong enough, because it reduces the chance of overfitting to a single opponent.

The `coach-neural` harness supports:

```powershell
npx tsx scripts/coach-neural.ts --accept-opponent league --gate-seeds 3
```

When the current neural model becomes reliably stronger, gradually shift training and gates away from traditional strategy and toward neural league opponents.

## Useful Commands

Install dependencies:

```powershell
npm install
```

Run the browser dev server:

```powershell
npm run dev
```

Run all tests:

```powershell
npm test
```

Build production assets:

```powershell
npm run build
```

Run sparse-reward PPO self-play smoke training:

```powershell
npx tsx scripts/train-policy-gradient.ts `
  --input public/models/neural-best.json `
  --output training-runs/neural-pg-candidate.json `
  --metrics-output training-runs/neural-pg-candidate-metrics.json `
  --seed 20260502 `
  --matches 24 `
  --frames 240 `
  --epochs 4 `
  --batch-size 64 `
  --learning-rate 0.0045 `
  --ppo-clip 0.2 `
  --temperature 1.12 `
  --discount 0.992 `
  --start-state-mode outcome-curriculum
```

Build the Rust trainer:

```powershell
cargo build --release --manifest-path trainer-rust/Cargo.toml
```

Run sparse-reward PPO self-play through the native Rust sampler/trainer:

```powershell
npx tsx scripts/train-policy-gradient.ts `
  --native `
  --input public/models/neural-best.json `
  --output training-runs/neural-pg-rust-candidate.json `
  --metrics-output training-runs/neural-pg-rust-candidate-metrics.json `
  --seed 20260502 `
  --matches 240 `
  --frames 240 `
  --epochs 4 `
  --batch-size 128 `
  --learning-rate 0.0045 `
  --ppo-clip 0.2 `
  --temperature 1.12 `
  --discount 0.992 `
  --start-state-mode outcome-curriculum
```

Run coach harness with RL cycles and league gate:

```powershell
npx tsx scripts/coach-neural.ts `
  --input public/models/neural-best.json `
  --output training-runs/neural-rl-gated-candidate.json `
  --cycles 0 `
  --rl-cycles 1 `
  --rl-matches 24 `
  --rl-frames 240 `
  --rl-epochs 4 `
  --rl-batch-size 64 `
  --rl-learning-rate 0.0045 `
  --rl-ppo-clip 0.2 `
  --rl-temperature 1.12 `
  --rl-discount 0.992 `
  --rl-start-state-mode outcome-curriculum `
  --accept-opponent league `
  --gate-seeds 3
```

Run coach harness with Rust native RL cycles and league gate:

```powershell
npx tsx scripts/coach-neural.ts `
  --input public/models/neural-best.json `
  --output training-runs/neural-rl-rust-gated-candidate.json `
  --cycles 0 `
  --rl-cycles 1 `
  --rl-native `
  --rl-matches 240 `
  --rl-frames 240 `
  --rl-epochs 4 `
  --rl-batch-size 128 `
  --rl-learning-rate 0.0045 `
  --rl-ppo-clip 0.2 `
  --rl-temperature 1.12 `
  --rl-discount 0.992 `
  --rl-start-state-mode outcome-curriculum `
  --accept-opponent league `
  --gate-seeds 3
```

Example runtime gate comparison:

```powershell
@'
import { readFileSync } from 'node:fs';
import { loadWeightsPayload } from './scripts/coach-neural';
import { evaluateRuntimePolicy } from './src/ai/policyGate';

for (const [name, path] of [
  ['best', 'public/models/neural-best.json'],
  ['candidate', 'training-runs/neural-pg-candidate.json']
] as const) {
  const weights = loadWeightsPayload(readFileSync(path, 'utf8'));
  let gf = 0, ga = 0, score = 0, win = 0, bp = 0;
  for (const seed of [19, 31, 43, 57, 71]) {
    const r = evaluateRuntimePolicy(weights, { seed, matches: 4, frames: 600 });
    gf += r.goalsFor;
    ga += r.goalsAgainst;
    score += r.score;
    win += r.winProxy;
    bp += r.ballProgress;
    console.log(`${name} ${seed}: goals=${r.goalsFor}-${r.goalsAgainst} score=${r.score.toFixed(1)} win=${r.winProxy.toFixed(3)} bp=${r.ballProgress.toFixed(3)}`);
  }
  console.log(`${name} TOTAL goals=${gf}-${ga} avgScore=${(score / 5).toFixed(1)} avgWin=${(win / 5).toFixed(3)} avgBp=${(bp / 5).toFixed(3)}`);
}
'@ | npx tsx -
```

## Development Rules

### General

- Keep runtime gameplay stable.
- Keep deterministic simulation behavior unless a physics change is intentional and tested.
- Prefer small, testable changes.
- Do not overwrite user replays or training artifacts.
- Do not commit generated builds, dependency folders, local logs, or large training outputs.
- Use `public/models/neural-best.json` only for accepted playable weights.
- Put experimental candidates under `training-runs/`.

### Testing

Before claiming a change is complete, run:

```powershell
npm test
npm run build
```

For narrow AI changes, also run relevant targeted tests first:

```powershell
npx vitest run tests/policyNetwork.test.ts tests/policyGradientTraining.test.ts tests/coachNeural.test.ts
```

If changing runtime policy behavior, run a deterministic gate comparison before replacing weights.

### Training Discipline

- Never treat training loss alone as model quality.
- Never accept a candidate just because it scored well in its own training episodes.
- Always compare to the current accepted model.
- Prefer league gates over single-opponent gates for accepted model replacement.
- Keep rejected candidates in `training-runs/` only for analysis.
- If a gate run times out, do not count it as pass or fail; rerun with a smaller or longer-timeout gate and document the result.

### Replay Discipline

Browser replay files are valuable training/evaluation data but may be large and user-specific. They should not be committed by default.

Expected replay filename pattern:

```text
browser-replay-*.json
```

If a replay becomes part of a public benchmark, move it into a documented benchmark folder intentionally and explain why it is safe to publish.

### Rust And Native Training

Rust is allowed for fast offline training and self-play simulation. The browser runtime does not need to be ported to Rust.

Preferred long-term split:

- Browser TypeScript: gameplay, UI, runtime policy inference.
- TypeScript tests: correctness and compatibility checks.
- Rust: high-volume self-play episode generation and training.
- Output format: compatible neural weight JSON loaded by the TypeScript runtime.

Native tools must preserve the current policy shape unless a coordinated migration is done:

- input count: 36,
- hidden layers: 64, 64,
- output count: 9,
- action mapping from `src/ai/policyActions.ts`.

## Git And Repository Hygiene

Commit source, tests, public accepted model weights, package manifests, and documentation.

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

The repository should be public only after checking that no private replay data, credentials, tokens, or large local artifacts are staged.

Recommended pre-publish check:

```powershell
git status --short
git check-ignore -v node_modules dist training-runs browser-replay-2026-04-30T07-50-03-445Z.json vite-dev.log
```

## Open Research Problems

The current PPO/self-play pipeline is functional but not yet stronger than the current accepted model in short training runs. Known next steps:

1. Port sparse-reward self-play simulation to Rust for many more episodes.
2. Add a league of frozen neural snapshots.
3. Track Elo-like or gate-score history per candidate.
4. Consider value-function or actor-critic learning to reduce sparse-reward variance.
5. Consider search-guided policy improvement for short tactical horizons, but keep search separate from hard-coded tactical labels.
6. Keep outcome-curriculum starts, but avoid turning them into hand-authored tactical rewards.
7. Measure stamina behavior through runtime commands, not only raw network predictions, because runtime post-processing may regulate commands.

The guiding principle remains: let the AI learn tactics from self-play outcomes as much as possible, and use gates to prevent regressions.
