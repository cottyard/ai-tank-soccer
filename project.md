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

Prefer self-play reinforcement learning over hand-authored tactical labels. Older supervised replay, curriculum labels, tactical rollouts, and traditional-strategy opponents have not produced promotable runtime policies in recent evidence; keep them only for regression tests, parity checks, and narrow diagnostics, not as mainline AI-improvement methods.

Current sparse-reward PPO supports:

- open, outcome-curriculum, own-goal-defense, corner-fight, loose-ball-contest, and mixed starts;
- runtime-like action execution in the Rust sampler;
- frozen opponent weights;
- traditional opponent mode for diagnostics/stabilization;
- weighted league opponent sampling from current, recent, historical, and traditional opponents;
- PPO clipping through old action probability;
- advantage normalization with `global`, `start-team-time`, or `learned` baselines;
- searchable open-start weighting inside `mixed` starts via `--open-start-ratio` / `--open-start-ratios`;
- trainable-only sample filtering when playing against frozen or traditional opponents.

Rewards should stay tied to outcomes: future goals for/against and final win/loss/draw. Start-state curriculum is acceptable only when it is evaluated through both standard and holdout gates; shaped tactical rewards should be separate, documented, and used sparingly.

## Method Audit

The recent plateau is not from a lack of tooling; it is from optimizing against weak or partial signals. Several runs improved one slice of behavior while failing the promotion gate, and earlier search ranked candidates before holdout or standard safety was fully represented.

Keep as mainline:

- Deterministic browser-runtime promotion gates. They are the only evidence that matches the playable model.
- Native Rust sparse-reward PPO with runtime action execution, `start-team-time` advantages, self-play opponents, and mixed starts as the accepted baseline recipe.
- Promotion-safe search infrastructure that evaluates standard and holdout gates together, searches explicit training seeds, and ranks standard-safe candidates before holdout-only gains.

Keep only as diagnostics:

- Open-start PPO. It consistently found a holdout seed `109` finishing gain, but every holdout-positive candidate still had standard score regression.
- Learned baseline and weighted league opponent sampling. The default learned+league promotion run regressed standard score and goals; keep these as searchable variants only when testing a specific hypothesis.
- Traditional opponents, supervised replay, curriculum labels, and tactical rollouts. They are useful for parity and regression coverage, but recent accepted progress came from sparse-reward runtime-gated PPO, not from hand-authored labels.
- Runtime hill climbing, output-bias probes, stamina-threshold tuning, and local black-box weight tweaks. They overfit quickly and should not consume mainline training budget.

Stop recommending:

- Bare default promotion runs that use learned baseline + league sampling without a new hypothesis.
- One-off learning-rate or epoch tweaks after the documented failures at `2026050210` and `2026050211`.
- Narrow clip/temperature sweeps around the current accepted recipe unless they are tied to a diagnosed gate seed failure.
- Searches with short `gate-frames` that do not produce goals; they rank noise, not soccer strength.

Current lesson: the accepted policy is near a local gate plateau. Useful future work should explain specific seed-level regressions before launching more grid searches. The immediate target is to make training updates behavior-visible under the runtime wrapper, then preserve any open-start holdout `109` gain without standard score loss.

## Development Principles

At the end of every work session, evaluate each training method and existing mechanism against the latest gate results, diagnostics, and implementation evidence. Keep methods that produce measurable progress, keep uncertain methods only as documented diagnostics, and remove or de-prioritize mechanisms that repeatedly fail to improve the accepted runtime policy.

每次完成工作后，去掉project.md中过时无用的内容，如果本轮工作ai没有提高，反思原因。并制定下一步计划。提交并push改动。

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

This trains a candidate from `public/models/neural-best.json`, runs the standard runtime gate and then the holdout runtime gate, writes `training-runs/neural-promotion-summary-s2026050208.json`, appends a compact entry to `training-runs/neural-promotion-history.jsonl`, and replaces `public/models/neural-best.json` only if both gates pass without meaningful goals or win-proxy regression. Use `--no-promote` for a dry run. Do not run the learned-baseline + league-sampling defaults as the next improvement attempt unless the recipe has first been changed to address the documented standard-gate regression.

Recent 2026-05-02 rejected promotion attempts:

| Seed | Variant | Candidate Standard Gate | Decision |
| --- | --- | --- | --- |
| `2026050208` | `learned` baseline + weighted `league` opponent, `lr=0.001`, `epochs=2` | goals `9-1`, `avgScore=265.482`, `avgWin=0.700`, `avgBp=0.275` versus current goals `11-1`, `avgScore=320.213`, `avgWin=0.725`, `avgBp=0.270` | Reject; de-prioritize this exact recipe until modified. |
| `2026050209` | `start-team-time` baseline + `self` opponent, `lr=0.001`, `epochs=2` | goals `10-1`, `avgScore=290.814`, `avgWin=0.675`, `avgBp=0.268` | Reject; closer on goals but still lower score/win proxy. |
| `2026050210` | same as above with `lr=0.0005` | goals `8-1`, `avgScore=238.851`, `avgWin=0.675`, `avgBp=0.293` | Reject; reducing learning rate alone made gate performance worse. |
| `2026050211` | same as above with `epochs=1` | goals `8-1`, `avgScore=237.964`, `avgWin=0.675`, `avgBp=0.282` | Reject; reducing epochs alone made gate performance worse. |

Internet research via `http://127.0.0.1:10808` found two immediately testable ideas from primary sources: PPO's clipped surrogate is intended to support multiple minibatch epochs but remains hyperparameter-sensitive ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)); Population Based Training searches hyperparameters and schedules under a fixed budget instead of relying on one fixed recipe ([Jaderberg et al. 2017](https://arxiv.org/abs/1711.09846)). In this project, blind PBT/grid expansion has already exposed trade-offs without promotion. Use search only after defining a seed-level hypothesis and keep MuZero-style learned-model planning ([Schrittwieser et al. 2019](https://arxiv.org/abs/1911.08265)) out of the near-term path.

League opponents can include the current accepted model, extra snapshots, and a low-weight traditional stabilizer for diagnostics only:

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
  --training-seeds 2026050212,2026050213 `
  --matches 240 `
  --frames 180 `
  --gate-matches 2 `
  --gate-frames 360 `
  --learning-rates 0.001,0.0008,0.0006 `
  --epochs-list 1,2 `
  --ppo-clips 0.08,0.12,0.16 `
  --temperatures 1.0,1.1 `
  --start-state-modes mixed,open `
  --open-start-ratios 0.2,0.35,0.5 `
  --advantage-baselines start-team-time,learned `
  --opponent-modes self,league
```

This writes ranked short-run candidates under `training-runs/policy-gradient-search-s<seed>/`, writes a JSON summary, appends `training-runs/policy-gradient-search-history.jsonl`, and does not promote weights. The search runner now supports explicit `--training-seeds`, searches `start-state-mode` / `open-start-ratio` / `advantage-baseline` / `opponent-mode` dimensions, evaluates both standard and holdout gates, and ranks promotion-safe standard candidates before holdout-only gains. `--open-start-ratios` applies only to `mixed` starts; non-mixed `open` variants are not duplicated by ratio. Use the best survivor as input to the full promotion loop before replacing `public/models/neural-best.json`.

Evaluate a search survivor through the full promotion gates without retraining it:

```powershell
npx tsx scripts/promote-policy-gradient.ts `
  --candidate-input training-runs/policy-gradient-search-s2026050217/v01-lr0p001-e1-clip0p12-t1p1.json `
  --summary-output training-runs/neural-promotion-search-survivor-summary-s2026050217.json
```

Search notes: a too-short `gate-frames=240` search produced no goals and no ranking signal, so short searches should keep the full `gate-frames=600` when possible. The `2026050217` two-variant search (`matches=120`, `frames=120`, full standard gate) found `lr=0.001`, `epochs=1`, `ppoClip=0.12`, `temperature=1.1` with standard delta `+0.447`, goals unchanged at `11-1`, and holdout unchanged at goals `11-0`, `avgScore=353.840`, `avgWin=0.750`, `avgBp=0.289`. The full `--candidate-input` promotion check rejected it because the holdout score did not improve. Keep this as a non-regressing survivor and useful search signal, not an accepted promotion.

The `search-policy-gradient` runner now evaluates both standard and holdout gates directly. A follow-up full-gate search on `2026050219` with a narrow grid over `ppo-clip={0.08,0.12,0.16}` and `temperature={1.0,1.1}` kept `lr=0.001`, `epochs=1`, and `start-team-time` / `self` fixed, but still produced no holdout score gains across all 6 variants. The best row matched the previous pattern: standard delta `+0.447`, holdout delta `0.000`, goals `11-1` standard and `11-0` holdout. That means the current narrow search surface is exhausted; future runs should widen the search space before burning more full-gate time.

Broadened full-gate search on `2026050220` (`matches=120`, `frames=120`) over `start-state-mode={mixed,open}`, `advantage-baseline={start-team-time,learned}`, and `opponent-mode={self,league}` found a useful but non-promotable trade-off. `open + start-team-time + self` with `lr=0.001`, `epochs=1`, `ppoClip=0.12`, `temperature=1.1` improved holdout to goals `13-1`, holdout delta `+19.319`, mostly by turning seed `109` from `1-0` to `3-0`, but regressed standard to goals `9-1`, standard delta `-55.322`. With promotion-safe ranking added, `2026050221` and `2026050226` showed the same split: temperature `1.0` can preserve the holdout `+19.319` signal with standard goals still `11-1` but standard score slightly negative (`-1.140` best observed), while temperature `1.05` restores standard delta `+0.447` and loses the holdout gain. Multi-seed search `2026050222..2026050225` reproduced the holdout-positive pattern for `open + start-team-time + self`, but no seed met the standard score gate. Treat open-start PPO as a promising diagnostic direction, not a promotion candidate yet.

On 2026-05-05, the trainer/search plumbing was extended so `mixed` starts can search an `openStartRatio` without switching all training to pure `open`. The TypeScript collector and Rust sampler both keep the original five-family `mixed` cycle when no ratio is provided; when a ratio is provided, only the open share changes and the remaining slots spread over outcome, own-goal-defense, corner, and loose-ball starts. On 2026-05-06, Rust stable GNU was installed locally through rustup using the `rsproxy.cn` mirror, `trainer-rust/target/release/soccer-policy-trainer.exe` was built, and native parity tests passed. The search CLI now also accepts PowerShell-split comma lists, so `--start-state-modes mixed,open` and an accidentally split `mixed open` both parse into the intended grid.

The first full-gate ratio diagnostic on seed `2026050602` (`matches=120`, `frames=120`, `lr=0.001`, `epochs=1`, `ppoClip=0.12`, `temperature=1.0`, `start-team-time`, `self`) compared `mixed` open ratios `0.2/0.35/0.5` against pure `open`. Best row was pure `open`: standard goals `12-1`, standard delta `+26.701`, holdout goals `11-0`, holdout delta `0.000`; full `--candidate-input --no-promote` check rejected it because holdout score did not improve. Ratio rows did not recover the seed `109` holdout gain: `0.35` and `0.5` matched the known safe standard `+0.447` / holdout `0.000` pattern, while `0.2` regressed standard by `-25.808`. Do not promote any `2026050602` candidate.

On 2026-05-06, runtime decision tracing was added through `scripts/trace-runtime-policy.ts`. It records policy argmax actions, tactical rollout rewrites, stamina conservation, critical-stamina regulation, and final action histograms without changing normal gate scoring. Tracing the rejected `2026050602` pure-open candidate explained why model improvement is hard: on standard seeds it gained one goal mostly through seed `19`, with only small final-action shifts (`+11` full-forward, `+7` forward-left, `-24` coast-right over 2000 decisions), while on holdout seeds the final action histogram was exactly unchanged across all 2000 runtime decisions. Policy argmax changed by only two holdout decisions (`-1` action 5, `+1` action 7), and the tactical/stamina wrapper erased even that. This means many PPO updates are too small or too misaligned to cross the runtime behavior threshold created by argmax action selection, tactical rollout overrides, and stamina guards. More training on the same objective can produce real weight deltas that are invisible in browser behavior.

Later on 2026-05-06, trace comparison was promoted into shared gate/search plumbing. `compareRuntimeTraces` now reports distribution-change counts/rates for policy, tactical, and final actions plus wrapper-rate deltas, and `scripts/search-policy-gradient.ts --trace-gate` attaches standard/holdout trace deltas to each row and history entry. Trace-gated search still ranks promotion-safe standard gates first, then rejects standard-trace-unsafe rows ahead of behavior-visible holdout ties when standard stamina-stop or critical-stamina regulation rates increase. Use `--trace-gate` for future short searches intended to find runtime-visible candidates; unchanged holdout final-action histograms are now a direct rejection signal, and standard wrapper-rate regressions should block ranking gains even if holdout action changes look promising.

Trace-gated search `2026050603` directly tested the previous open-start hypothesis with `open + start-team-time + self`, `seed=2026050222`, `matches=120`, `frames=120`, full `gate-frames=600`, `ppoClip=0.12`, and `temperature={1.0,1.05}`. The best `temperature=1.05` row had standard delta `+0.893` with only `0.003` standard final-action change rate and no standard wrapper regression, but holdout delta was exactly `0.000` with holdout final-action change rate `0.000`; reject as holdout behavior-invisible. The `temperature=1.0` row produced holdout final-action change rate `0.038`, but regressed standard score by `-2.280`, holdout score by `-1.780`, and increased standard/holdout stamina-stop rates; reject as unsafe. This confirms trace-gated ranking is filtering the old temperature trade-off correctly and that no `2026050603` candidate should enter promotion.

Also on 2026-05-06, the native Rust PPO sampler gained runtime wrapper-survival diagnostics. Metrics now include `policyActionSurvival` with sampled, survived, changed, tacticalChanged, staminaConserved, criticalRegulated, and survivalRate fields, plus a `--runtime-survivors-only` training filter that keeps only trainable runtime samples where the sampled policy action survived tactical/stamina wrapping. A smoke run on open starts (`seed=2026050604`, `matches=40`, `frames=120`, `epochs=0`) showed survival `1327/1600 = 0.829`, with `273` changed actions, mostly tactical rollout (`261`) and some stamina conservation (`17`). This supports the diagnosis that PPO is partly optimizing executed wrapper actions rather than policy decisions that will survive in the browser.

The first survivors-only trace search `2026050605` used the same open-start recipe as the unsafe `temperature=1.0` row but added `--runtime-survivors-only`. It trained on `3890` survivor samples out of `4800` trainable runtime decisions (`survivalRate=0.810`) and still regressed standard score by `-2.460` and holdout score by `-1.780`, with holdout final-action change rate `0.038` and increased stamina-stop rates. This means filtering out wrapper-rewritten samples alone is not enough to recover the holdout `109` gain safely. Keep `--runtime-survivors-only` as a diagnostic/search dimension, not a default promotion recipe.

When using `--candidate-input`, the promotion loop reads the candidate weight metadata for seed, baseline, opponent mode, epochs, batch size, learning rate, clip, temperature, open-start ratio, discount, start mode, and action mode. This keeps promotion summaries and history tied to the actual search candidate instead of the promotion loop's default training recipe.

When internet research is needed from this environment, use the local HTTP proxy `http://127.0.0.1:10808`.

## Next Work Plan

1. Do not promote or full-gate-check `2026050603` or `2026050605`; neither improved the accepted AI. The reason is now clearer: behavior-visible holdout changes either disappear at `temperature=1.05` or arrive with standard/holdout score and stamina-wrapper regressions at `temperature=1.0`, and survivor-only filtering does not fix that trade-off.
2. Use the new `policyActionSurvival` metrics before future searches. If a recipe has high wrapper rewrite rates, compare candidate behavior against survivor counts and wrapper categories before spending full-gate time; changed holdout actions are only useful if standard score and stamina/critical wrapper rates remain safe.
3. Next implementation target: add a sampler-side diagnostic that splits advantages/returns by `survived` versus `changed` runtime decisions and by wrapper category. This should show whether rewritten tactical/stamina samples carry different return signs or advantage magnitudes before changing the objective again.
4. If the split diagnostic shows changed samples dominate useful returns, test a more precise objective than `--runtime-survivors-only`: for example down-weight wrapper-changed samples or train the policy on sampled actions only when the wrapper would not override them, while preserving gate evaluation through `--trace-gate`.
5. Keep learned baselines and weighted league sampling available, but treat them as variants to search only after the behavior-visibility problem is addressed. Keep runtime-action Rust parity covered by tests whenever `neuralStrategy`, stamina regulation, tactical rollout, or physics changes.

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
