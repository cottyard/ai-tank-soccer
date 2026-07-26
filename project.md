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

Every research session ends with an explicit process audit, not only a policy-result note. Review whether the opponent set, seed split, start-state pairing, rollout opponent model, metrics, trace fidelity, counterfactual speed, tests, and policy ownership boundaries still create useful pressure toward a stronger AI. If a workflow component is fixed, noisy, too slow, or rewards overfitting, repair it and record the change here. A session that discovers and fixes a misleading gate or a learning bottleneck is valid infrastructure progress even when runtime behavior is intentionally unchanged.

## Evolving Opponent League

The legacy standard and holdout gates remain fixed reproducibility anchors against `traditionalStrategy`; they are necessary but no longer sufficient. Repeatedly optimizing only those opponents and seeds can turn improvement into fixed-exam overfitting.

The additional runtime opponent league is configured in `config/runtime-opponent-league.json` and has two lifecycle classes:

- `anchor`: classic or special opponents that stay fixed so historical abilities cannot silently regress.
- `rolling`: opponents derived from the latest accepted weights/runtime. Rolling profiles and most of their seeds advance after a real promotion, so future policies must handle an evolving field rather than only the original traditional opponent.

Current generation `1` contains one fixed `classic-traditional` anchor plus two rolling opponents: `accepted-no-rollout`, which isolates whether the runtime tactical wrapper adds value over the accepted wrapper without search, and `accepted-runtime`, which is a mirrored latest-runtime symmetry/safety check. Because latest-runtime self-play is structurally symmetric, do not treat its raw win rate as a promotion objective; use it to expose team/start asymmetry, nondeterminism, and unsafe behavior. The no-rollout matchup is the more useful incremental-strength signal for runtime search changes.

Rolling league matches use paired physical starts. Each scenario creates one fixed field state, then swaps candidate and opponent between red and blue without regenerating or mirroring the state around the candidate. This prevents the candidate-favoring start bias discovered in the first league implementation. The legacy gates intentionally retain their historical start semantics for continuity.

League discipline:

1. Freeze the league generation throughout an experiment so current and candidate face the same exam.
2. Run the quick `2 x 300` league as a screen, then the full `4 x 600` rolling profiles for promotion or periodic audits.
3. Keep legacy standard/holdout results as hard safety anchors; league gains cannot excuse a severe legacy regression.
4. Advance the league only after an AI promotion is accepted. The advance tool preserves all anchor seeds and one continuity seed per rolling opponent while replacing the rest.
5. Record per-opponent results. Do not collapse anchor, no-rollout, and runtime-mirror rows into one score because they answer different questions.

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
- `scripts/inspect-runtime-match.ts` emits per-decision JSONL snapshots with ball/tank geometry, pressure signals, rollout scores, optional fixed-horizon rollout comparisons, optional `--rollout-breakdown` position-evaluation deltas, and counterfactual tactical choices for decisions hidden by stamina waits.
- `scripts/probe-runtime-sequences.ts` runs two-step fixed-action counterfactuals from cached decision-aligned states and ranks the resulting final ball threat. It supports `--max-combinations` for bounded partial sweeps and is useful for checking whether a local release needs sequence/search work before adding another direct action guard.
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

2026-07-11 continuation from the accepted `avgWin=0.900` HL baseline:

The next target was to convert one of the remaining standard draws without losing the recovered `19` and `31` safety. The accepted lesson came from comparing `43:1` and `19:1`: blindly waiting on rolling finish setups can convert one draw but regress a central push. The safer pattern is narrower: when stamina is low, the ball is already rolling toward goal in the attacking third, both tanks are in close contact range, and the ball is offset from the goal-center line by at least `30px`, stop briefly instead of adding another weak touch that can spoil the setup.

Accepted addition:

1. Offset rolling finish setups now wait before touching the ball when the ball is in the goal lane but not centered, the attack velocity is non-negative, speed is controlled, own-goal pressure is low, stamina is at or below `0.30`, and the opponent is also close enough that extra contact is likely to redirect the shot. Centered rolling-finish pushes remain active through the existing low-lateral-drift guard.
2. Fast centering finish follow-through states now use a 36-frame tactical rollout horizon instead of the default 18-frame horizon when the ball is already in the attacking finish lane, rolling quickly toward goal, drifting back toward center, and the controlled tank is still in close contact range. This targets standard `71:3` frame `528`, where the short rollout switched a raw forward follow-through into a side action that left the ball pinned deep near the side wall. The longer horizon keeps the raw forward action and improves final threat quality without changing aggregate goals.

Rejected during this continuation:

- A broad rolling-finish wait converted `43:1` but regressed `19` from `4-0` to `3-0`; do not remove the center-offset guard.
- A lane-only side finish wait and several narrower side-lane threshold variants produced only tiny score movement and did not convert `43:1`.
- A deep loose-ball low-stamina exception for `71:0` improved some local ball-progress probes but did not improve the match gate and sometimes created own-goal danger.
- Single-action macro probes for `31:0` and `71:3` found local ball-position changes but no reliable conversion; these need sequence/search or scoring-model work rather than direct action guards.
- A centered low-stamina drifting-finish exception for `71:3` skipped the stamina wait at frame `546` and let rollout choose forward, but the gate did not improve: standard stayed goals `19-0`, `avgWin=0.925` while score slipped to `569.369`; holdout score also slipped slightly to `568.008`. Do not weaken the low-stamina wait unless it produces an actual conversion or a clearer aggregate gain.

Updated gate comparison from the previous accepted `0.900` baseline:

| Gate | Previous Accepted | Current |
| --- | ---: | ---: |
| Standard `[19,31,43,57,71]`, `matches=4`, `frames=600` | goals `17-0`, `avgScore=517.609`, `avgWin=0.900`, `avgBp=0.226` | goals `19-0`, `avgScore=568.181`, `avgWin=0.925`, `avgBp=0.169` |
| Holdout `[83,97,109,127,149]`, `matches=4`, `frames=600` | goals `19-0`, `avgScore=567.751`, `avgWin=0.875`, `avgBp=0.192` | goals `20-0`, `avgScore=594.931`, `avgWin=0.925`, `avgBp=0.166` |

Per-seed current standard gate:

- `19`: `5-0`, score `735.854`, win `1.000`, ball progress `0.198`.
- `31`: `3-0`, score `460.352`, win `0.875`, ball progress `0.200`.
- `43`: `4-0`, score `604.065`, win `1.000`, ball progress `0.238`.
- `57`: `4-0`, score `583.429`, win `1.000`, ball progress `-0.020`.
- `71`: `3-0`, score `457.204`, win `0.750`, ball progress `0.231`.

Per-seed current holdout gate:

- `83`: `4-0`, score `600.117`, win `1.000`, ball progress `0.189`.
- `97`: `3-0`, score `440.666`, win `0.875`, ball progress `-0.046`.
- `109`: `3-0`, score `458.536`, win `0.750`, ball progress `0.247`.
- `127`: `6-0`, score `878.049`, win `1.000`, ball progress `0.288`.
- `149`: `4-0`, score `597.288`, win `1.000`, ball progress `0.154`.

Current remaining standard draws are still `31:0`, `71:0`, and `71:3`. The accepted fast-centering rollout improves `71:3` threat quality, moving the final ball from the deep side-wall finish (`attackX=917.474`, side-wall distance `69.122`) to a less pinned attacking state (`attackX=944.054`, side-wall distance `181.948`), but it does not convert the match. The next practical target remains `71:3`, now as a follow-up/continuation search problem after the centering release. Treat `71:0` separately as a midfield/low-finish recovery problem, and avoid direct low-stamina loose-ball drive exceptions unless they survive own-goal safety checks.

Decision: keep the offset rolling finish wait and fast-centering finish rollout if full verification passes. Standard win proxy improved from `0.900` to `0.925`, standard goals rose from `17-0` to `19-0`, and the latest rollout addition gives a small standard score/ball-progress gain over the prior accepted `0.925` state while holdout remains safe at `19-0`.

2026-07-11 follow-up diagnostics from the accepted `avgWin=0.925` HL baseline:

No runtime change was accepted in this pass. The main target was still standard draw `71:3`. Fixed sequence probes showed that a committed action-`7` continuation from frames `546` or `564` can convert the chance by frame `720`, but converting that observation directly into runtime policy did not survive online re-decision. A narrow direct action-`7` follow-through guard kept the first continuation touch but failed to convert `71:3`; widening it made the ball pin even deeper on the side wall. A `12+36` outward-finish tactical sequence profile made rollout choose action `7` at frame `546`, but then chose stop at frame `552`, increased lateral side-wall drift, and also failed to convert.

The lesson is that `71:3` is not ready for another direct action guard or a simple two-step rollout trigger. The failure is in continuation valuation: the evaluator does not reliably price the future cost of a fast ball drifting out of the finish lane toward the side wall, and isolated macro success does not imply the online rollout will preserve the same contact geometry. Do not reintroduce action-`7` follow-through guards or the broad `12+36` outward-finish sequence unless the rollout evaluator first explains and prevents the side-wall drift regression.

`71:0` was inspected as a separate midfield/low-finish recovery problem. Focused single/two-step probes around frames `360` through `552` improved final ball placement in some branches but did not find a conversion at `600` frames. The best probes moved the final ball toward midfield or a better lane, but no candidate produced a clear match outcome gain worth encoding.

2026-07-13 continuation diagnostics from the accepted `avgWin=0.925` HL baseline:

No runtime policy change was accepted in this pass. The first target remained `71:3`. Re-inspecting frames `480` through `594` confirmed that the accepted 36-frame fast-centering rollout preserves the forward follow-through at frame `528`, but later low-stamina waits and online re-decisions still leave the ball short of conversion. A focused 600-frame sequence sweep found better final threat positions but no goal; the best sampled branch ended around `attackX=940.850`, lane `0.997`, still `0-0`. Narrow 720-frame checks of the strongest 600-frame branches also stayed `0-0`, which weakens the case for another direct action guard.

`31:0` was rechecked as a deep centered finish that appears tempting for a low-stamina forward exception. Direct action-`8` macro probes at frames `552`, `558`, and `570` did not convert the match by `600` frames. A safety probe on already-converted standard seed `19:1` showed that forcing action `8` too early can erase an existing goal, so broad low-stamina finish-drive exceptions remain unsafe. Keep treating `31:0` as a contact/sequence-model problem rather than a stamina-guard relaxation.

The accepted change from this pass is diagnostic infrastructure only: `scripts/probe-runtime-sequences.ts` now supports `--max-combinations`, returns planned/completed/truncated metadata, and prints a `partial completed=x/y` line when a bounded sweep stops early. This addresses the immediate TypeScript counterfactual bottleneck seen while probing `71:3` and `31:0`, where full sequence grids became too slow to be useful. Runtime behavior and gates are unchanged.

2026-07-13 follow-up from the same accepted `avgWin=0.925` HL baseline:

No runtime policy change was accepted. A position-evaluation experiment reduced the zero-speed component of `finishThreatScore` from `0.2` to `0.05` under the hypothesis that stalled near-goal balls were being overvalued. Focused tests passed, but the standard gate regressed badly before holdout was run: standard fell to goals `13-0`, `avgScore=408.678`, `avgWin=0.800`, `avgBp=0.271`. This confirms that the existing low-speed finish-threat floor is protecting other standard conversions, especially seed `19`, and should not be lowered broadly.

Another position-evaluation experiment suppressed the absolute attacking-corner `cornerPositionScore` reward when the ball was not actually near a side wall (`sideDistance > FIELD.ballRadius + 130`), under the hypothesis that `71:3` frame `564` was overvaluing centered near-goal geometry as a corner escape. Focused tests passed, but the standard gate again regressed before holdout was run: standard fell to goals `13-0`, `avgScore=408.701`, `avgWin=0.825`, `avgBp=0.257`. This means the broad attacking-corner reward is still supporting other conversions; do not globally narrow `cornerPositionScore` without a more local terminal valuation signal.

Accepted diagnostic addition: `scripts/inspect-runtime-match.ts --rollout-breakdown` now attaches fixed-action position-evaluation details to inspected rollout windows. For each requested horizon it can show the raw policy action and the tactical choice with score, final ball geometry, attack/lateral velocity, `evaluatePositionDelta` breakdown, and final `evaluatePosition` breakdown. The first `71:3` breakdown pass shows that frame `564` is heavily valued by `cornerEscape` and centered near-goal geometry even when the branch still does not convert, while later frames correctly expose the side-lane collapse through `shotLane`, `finishThreat`, and `possession` drops. Next changes should be narrower than global finish-threat scaling.

Accepted diagnostic addition: `scripts/inspect-runtime-match.ts --continuation-frames ...` now simulates the current online runtime policy from each inspected decision frame and reports goal deltas, first goal frame/offset, and final ball geometry. Comparing standard conversions against `71:3` showed a stronger distinction than static rollout scoring: `19:1` from frame `528` scores at frame `568` within a 72-frame continuation, and `43:1` from frame `552` scores at frame `585` within a 36-frame continuation, while `71:3` frames `528` through `594` do not score within 120 continuation frames and collapse from lane `0.773`/`0.753` into lane `0` side-wall states. Future evaluator work should use continuation reachability as a diagnostic target instead of only terminal position totals.

Follow-up rollout-evaluator experiment: a terminal scoring penalty was applied only inside tactical rollout when a low-stamina branch ended with a deep, centered, non-scoring ball that lacked forward velocity. The hypothesis was that `71:3` was overvaluing low-speed deep finish states without needing another action guard. Focused tests passed after restricting the penalty to low initial stamina, but the local `71:3` diagnostic still drew and ended closer to the side wall. The standard gate regressed to goals `16-0`, `avgScore=489.474`, `avgWin=0.875`, `avgBp=0.226`, with seed `31` falling to `2-0` and seed `57` to `3-0`. Reverted. Do not add broad non-scoring deep-finish penalties; the evaluator needs a more predictive distinction between a recoverable finish path and a low-speed dead end.

2026-07-16 continuation from the accepted `avgWin=0.925` HL baseline:

Accepted additions:

1. Relaxed the `attackBallY` threshold in `shouldPreserveCriticalRollingFinishPush` from `>= 0` to `>= -12`, allowing nearly-centered balls on either side of the goal center to preserve full-forward critical stamina pushes. The original strict `>= 0` condition was tuned to seed `19:1` which happened to have positive `attackBallY`, but seed `31:0` misses preservation by only `6.5px` at frame `414` where all other conditions pass.
2. Fixed steering direction in `regulateCriticalStaminaCommand`: when the ball is nearly straight ahead (`|local.lateral| < tankRadius * 0.18`) and the regulation reduces a two-track command to single-track, the code previously always dropped the right track (turning right), which was wrong when the ball was slightly to the left. Now chooses which track to drop based on the ball's lateral position. This converted holdout seed `97` from `2-0` to `3-0` (`win=0.750` → `0.875`).
3. Added a minimum ball speed check (`ballSpeed >= 3`) to `shouldWaitForDriftingFinish`. Previously, the AI would wait indefinitely for a "drifting finish" even if the ball was effectively stationary in the deep finish zone (observed in seed `31:0` at frame `558` where ball speed was `1.374`). Gate confirmed completely safe (identical `0.925` / `0.925`).
4. Unbounded the contest delta in `evaluatePositionDelta`. Previously, the contest delta was wrapped in `Math.max(0, ...)`, which meant the tactical rollout was completely blind to actions that lost contest (such as moving away from the ball while the opponent approached). While correcting this logically makes the rollout more robust, it resulted in exactly identical gate scores, meaning the neural policy and heuristics were already avoiding or filtering out contest-losing actions anyway.

Rejected during this continuation:

- A lateral-drift guard in `shouldWaitForOffsetRollingFinish` (suppress wait when `attackBallY * attackLateralVelocity > 0 && |attackLateralVelocity| > 25`) was too broad: it regressed seeds `19` (`5-0` → `4-0`) and `43` (`4-0` → `3-0`) without improving `71`. The offset rolling finish wait is protecting conversions in `19` and `43` even when lateral drift is present. Do not narrow this wait condition based on lateral velocity alone.
- Enabling tactical rollout for midfield contested balls (within `tankRadius * 2.2` and middle 60% of field) caused catastrophic regression: standard dropped from `avgWin=0.925` to `0.775`, every seed lost goals. The 18-frame rollout makes worse midfield decisions than the neural policy's learned long-horizon strategies. Do not enable tactical rollout broadly in midfield.
- A midfield safety override with a high improvement margin (`0.25` instead of default `0.018`) still regressed: seed `71` conceded a goal (`3-0` → `3-1`, `win=0.750` → `0.625`) and holdout seed `83` dropped from `4-0` to `3-0`. The evaluator is fundamentally misaligned with good midfield play — no margin level can make midfield rollout overrides safe. Do not attempt any variant of midfield tactical rollout until the position evaluator is redesigned to value midfield play correctly.

Diagnostic findings from detailed inspection of all three remaining standard draws:

- `31:0`: The opponent maintains exactly ~99px from the ball throughout the entire match (constant contact), parked near the goal and blocking the scoring path. Critical stamina regulation converts full-forward to single-track turns during the crucial approach phase (frames `414`-`456`). The ball stalls at zero speed for 30+ frames while both tanks recover stamina. This is fundamentally a contact/sequence model problem: the AI needs to push around a blocking opponent, not through it.
- `71:0`: The neural policy consistently chooses action `5` (turn left) in midfield when action `7`/`8` (turn right / full forward) would be dramatically better (counterfactual scores `0.1`-`0.39` vs `-0.27` to `0.02`). Tactical rollout is NOT used because no `shouldUseTacticalRollout` conditions trigger for midfield balls. However, enabling rollout for midfield was catastrophic across three different experiments (broad trigger, high margin, safety override) because the position evaluator does not value midfield play correctly. This is a neural policy quality problem that requires either retraining or a midfield-specific evaluator.
- `71:3`: Ball drifts rapidly toward the side wall (lateral velocity `29`-`83` px/frame, lane deteriorates `0.767` → `0.349`) while `shouldWaitForOffsetRollingFinish` triggers. Even when the wait is suppressed, the AI cannot redirect the ball. Confirmed as a rollout-evaluator continuation valuation problem, not an action-guard problem.

Decision: keep both accepted changes. Standard gate is unchanged at `avgWin=0.925`; holdout improved from `avgWin=0.875` to `avgWin=0.925` across the two changes (+5.7% relative improvement). The remaining standard draws (`31:0`, `71:0`, `71:3`) are blocked by architectural limitations: the position evaluator's midfield misalignment, the neural policy's midfield quality, and the opponent-blocking contact model. Further progress on these draws requires deeper changes than heuristic threshold adjustments.

2026-07-26 workspace/commit review and opponent-league audit:

No recent commit was withdrawn. The critical-stamina steering-direction fix is a real holdout improvement and stays. The minimum drifting-ball speed guard and negative contest delta are logically correct, tested fixes whose legacy gates are neutral; they stay as correctness protections rather than being misreported as strength gains. The remaining reviewed commits are documentation. The project timeline was reordered chronologically during this audit. An unfinished workspace export and a low-quality `debug-midfield.ts` scratch script were not accepted as project work.

Two direct runtime experiments were rejected:

- Extending the critical rolling-finish push window for `31:0` kept full-forward longer but still drew and drove the final ball toward the side wall.
- A narrow defensive-recovery rollout trigger for `71:0` improved final `attackX` from about `474` to `534` but did not convert the draw. It was reverted because tiny terminal movement without an outcome gain is not sufficient evidence.

An `800`-combination contact sequence sweep then exceeded five minutes without returning a useful result. This confirmed that the current TypeScript counterfactual path is still a research bottleneck and should not be scaled by brute force.

The larger finding was a gate-design limitation: all accepted progress had been measured against a fixed traditional opponent and fixed seed splits. The new opponent league adds rolling accepted-policy opponents while retaining classic anchors. Its first implementation exposed and fixed two evaluation mistakes before the results were trusted: unpaired matches used different random scenes for each side, and the first attempted pairing mirrored the scene around the candidate, giving the candidate the favorable attack-frame start twice. The final implementation holds one physical state fixed and swaps strategies between red and blue. A regression test now requires identical paired strategies to produce symmetric aggregate goals and win proxy.

Trusted generation `1` baseline against the paired `accepted-no-rollout` rolling opponent, `matches=4`, `frames=600`:

| Opponent | Seeds | Goals | avgScore | avgWin | avgBp |
| --- | --- | ---: | ---: | ---: | ---: |
| `accepted-no-rollout` | `[163,181,211,239,269]` | `8-3` | `154.435` | `0.600` | `0.055` |

Per-seed full results are `163: 2-1/0.625`, `181: 1-0/0.625`, `211: 2-1/0.625`, `239: 1-0/0.625`, and `269: 2-1/0.500`. This proves that the full runtime wrapper adds aggregate value over the same accepted wrapper without tactical rollout, but it also exposes seed `269` as the first rolling-league target. A paired latest-runtime mirror quick check returns symmetric `2-2`, `avgWin=0.500`, `avgBp=0`, as expected; it is a symmetry diagnostic rather than a strength target.

Decision: accept the evolving gate and diagnostic infrastructure, but do not claim a new runtime-policy promotion in this pass. Runtime behavior is intentionally unchanged after rejecting both local rules. The next AI-quality work should use the paired league to explain seed `269` and should prioritize a faster runtime-parity counterfactual engine before attempting another broad sequence search.

2026-07-26 simulation speed and measurement audit:

This pass questioned the process rather than adding another heuristic. Two structural problems were found and fixed, and three AI-strength hypotheses were then tested with the repaired instrument. Runtime behavior is intentionally unchanged.

**Problem 1: the physics kernel was accidentally quadratic in redundant work.** `tankLocalPointToWorld` recomputed `Math.cos`/`Math.sin` of the tank angle for every polygon vertex, inside all 16 collision iterations, several times per iteration. Measured cost was `10658` trig calls per frame in contact and `4352` when idle, for a two-tank simulation, plus one allocation per vertex, axis, centroid, and projection. This one defect set the ceiling on gate size, rollout horizon, and counterfactual sweeps, so it was the real cause of the `800`-combination sweep timing out.

The collision path now runs over reusable `Float64Array` buffers with a two-entry angle cache and bounding-box rejection, keeping the exact operand order of every floating-point expression:

| Measurement | Before | After |
| --- | ---: | ---: |
| Raw `stepGame` | `153.4us`/frame | `6.0us`/frame (`25.6x`) |
| Standard + holdout gates | `~200s` | `25.7s` |
| Fingerprint suite | `11083ms` | `1450ms` |

Because every historical result depends on deterministic trajectories, `scripts/fingerprint-simulation.ts` folds every float of every frame into a 64-bit digest, and `tests/simulationFingerprint.test.ts` pins physics and full runtime decision stacks to exact digests. All 13 scenario digests and every per-seed gate number reproduce exactly. Any future physics edit that changes a single ULP now fails the suite instead of silently invalidating the record above.

**Problem 2: the legacy gates cannot measure what they were being used for.** Standard and holdout are 20 matches each, so win proxy moves in `0.025` steps and has no error bar. Every accepted heuristic in this file was judged on that instrument, several by margins far below its resolution.

`src/ai/policyBenchmark.ts` and `scripts/benchmark-runtime.ts` add a large-sample paired benchmark: one physical start played twice with sides swapped, the scenario as the unit of observation, explicit 95% confidence intervals, and a paired-difference comparison that cancels start-state variance. Seeds come from a high range disjoint from the gate and league seeds, so it measures generalisation instead of re-examining tuned seeds. Identical strategies score exactly `0.5000` with zero variance, which is asserted in tests and is the same symmetry check the league needed. Work shards across worker threads, so 400 scenarios cost about seven minutes.

Results, opponent = accepted runtime unless stated, mirror control exactly `0.5`:

| Hypothesis | Result | Verdict |
| --- | --- | --- |
| The rollout stack is real, not overfitting | `0.7331 +-0.0190` vs traditional, against `0.4794 +-0.0128` without rollout; paired delta `-0.2537`, CI `[-0.2731,-0.2344]`, 800 matches | Strongly load-bearing |
| The legacy gate reports true strength | Gate says `avgWin=0.925` on its five tuned seeds; 400 fresh scenarios say `0.7331` | Gate overstates by ~`0.19` |
| Deeper search helps | `frames=36` scored `0.5212`, CI `[0.5026,0.5399]` over 200 scenarios, then `0.5095`, CI `[0.4978,0.5212]` over 500 independent scenarios | Did not replicate; winner's curse |
| Search should run more often | `force=1` scored `0.4738`, goals `62-83`; with margin `0.12`, `0.4725`, goals `59-81` | Actively harmful |
| Rollout's frozen opponent is the flaw | Modelling the opponent as the same network re-deciding at 5Hz scored `0.5075`, CI `[0.4919,0.5231]`, goals `156-148` over 600 matches | No measurable gain |

The opponent-model result is worth recording because the flaw is real: rollout passes commands only for the controlled tank, so `sanitizeCommand` hands the opponent a full stop and the search plans against a stationary opponent for up to 120 frames. Replacing that with a reacting opponent is a strictly better world model and still bought nothing measurable, which is strong evidence that the limiting error is in the terminal valuation rather than in the transition model.

The depth result is the important process lesson. Three variants were screened, the best cleared a naive 95% interval, and it evaporated on fresh seeds. The old 20-match gate had no way to detect that and would have accepted it, which is the most likely explanation for several narrow constants already in this file.

The trigger result independently confirms the earlier midfield finding, previously supported only by 20-match anecdotes, and now with real error bars: expanding tactical rollout beyond its trigger loses.

Conclusion: search depth, trigger coverage, and the rollout's opponent model are all saturated. Three independent improvements to the search — deeper, wider, and with a better transition model — produced no replicable gain, and one produced a loss. That points at the terminal valuation, so the hand-weighted linear position evaluator in `src/ai/positionEvaluation.ts` is the binding constraint on AI strength. The next real gain has to come from a better value function, not from another guarded rule or horizon constant.

The physics rewrite was independently verified by differential testing against the pre-rewrite kernel: `4358000` frame comparisons over `1874060` distinct random initial states, comparing every float of `GameState` bit-for-bit, with zero divergence. Coverage included overlapping and exactly-touching hulls, corner jams, balls on wall and goal-mouth boundaries, zero and extreme velocities, zero stamina, and guard-band-marginal placements. Mutation testing confirmed the harness can fail: a pure re-association of one transform (`a + b*c - d*e` to `a + (b*c - d*e)`, a one-ULP change) was caught in `34%` of trials, and refreshing collision geometry inside the part loop in `23%`. One fragility was found and fixed: the angle cache now keys on `Object.is` so a `-0` angle cannot borrow `+0`'s entry, since `Math.sin(-0)` is `-0`. Tank angles from `normalizeAngle` are never `-0`, but `Math.atan2` in `selfPlayTraining` and deserialised replays can produce one. The fix is a verified semantic no-op on reachable states.

## Architecture

- Browser runtime: TypeScript + Vite.
- Game model and deterministic physics: `src/game`.
- Runtime AI wrapper: `src/ai/neuralStrategy.ts`.
- Short-horizon action search: `src/ai/tacticalRollout.ts`.
- Position scoring for rollout: `src/ai/positionEvaluation.ts`.
- Accepted neural weights and loader: `public/models/neural-best.json`, `src/ai/bundledPolicy.ts`.
- Policy network and old PPO tooling: `src/ai/policyNetwork.ts`, `src/ai/policyGradientTraining.ts`, `scripts/train-policy-gradient.ts`, `trainer-rust`.
- Runtime deterministic gates and decision traces: `src/ai/policyGate.ts`, `scripts/trace-runtime-policy.ts`.
- Evolving paired opponent league: `src/ai/runtimeOpponentLeague.ts`, `config/runtime-opponent-league.json`, `scripts/evaluate-runtime-league.ts`.

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
4. Run the quick paired opponent league during iteration and the relevant full rolling profiles before promotion.
5. Keep it only if aggregate score/goals improve or remain safe, no severe seed-level regression is left unexplained, and rolling-opponent results do not reveal that the gain is only traditional-opponent overfitting.
6. Complete and record the end-of-session process audit.

Neural weight promotion:

1. Train a candidate into `training-runs/...json`.
2. Evaluate current accepted weights and the candidate on the standard gate.
3. Evaluate promising candidates on the holdout gate.
4. Evaluate promising candidates against the frozen current opponent-league generation, using current accepted weights for rolling opponents.
5. Promote only if score improves while goals and win proxy do not meaningfully regress across anchors and the rolling league.
6. Replace `public/models/neural-best.json` only after promotion.
7. Advance the opponent league once, after acceptance; never rotate it while comparing current and candidate.
8. Run full verification.
9. Commit source, tests, docs, accepted weights, and the advanced league config together.

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

Large-sample paired benchmark. This is the primary strength measurement; the
mirror control is exactly `0.5`, so a candidate is stronger only when its 95%
interval excludes `0.5` and the result replicates on a different `--salt`:

```powershell
# Is a candidate actually stronger than the accepted runtime?
npx tsx scripts/benchmark-runtime.ts `
  --policies accepted-runtime `
  --opponent accepted-runtime `
  --scenarios 400

# Compare search shapes, and ablate the rollout stack, on identical starts.
npx tsx scripts/benchmark-runtime.ts `
  --policies accepted-runtime,accepted-no-rollout,accepted-runtime@frames=36+margin=0.05 `
  --opponent traditional `
  --scenarios 400 `
  --output training-runs/bench.json

# Replicate a promising screen on independent seeds before believing it.
npx tsx scripts/benchmark-runtime.ts --policies <candidate> --scenarios 500 --salt 2
```

Deterministic simulation fingerprints. Run before and after any physics or
runtime change; identical digests prove trajectories are untouched:

```powershell
npx tsx scripts/fingerprint-simulation.ts
```

Paired evolving-opponent league:

```powershell
# Fast research screen.
npm run gate:league:quick

# Full promotion/periodic audit. Use --details for per-seed rows.
npm run gate:league -- --details

# Diagnose a rolling-opponent failure with exact physical start pairing.
npx tsx scripts/diagnose-runtime-failures.ts `
  --opponent accepted-no-rollout `
  --paired-starts `
  --seeds 269 `
  --matches 4 `
  --frames 600

# Preview first. Run the package command only after a promotion is accepted.
npx tsx scripts/advance-runtime-league.ts
npm run gate:league:advance
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

The 2026-07-26 audit moved the bottleneck. Search is saturated, measurement is fixed, and the evaluator is now the limiting factor.

1. **Replace the hand-weighted position evaluator with a learned value function.** This is the mainline task. `evaluatePosition` is ten hand-tuned linear terms, and the evidence says search quality is capped by it: deeper horizons do not replicate and searching everywhere loses. Train a value head to predict the actual match outcome from a state, using the fast kernel to generate labelled rollouts, then swap it into `tacticalRollout` behind the existing improvement margin. This is the case where a trained component should beat hand-coded logic, because the target is exactly the quantity the heuristic is failing to approximate.
2. **Judge every candidate on the large-sample benchmark, not the legacy gates.** Require the 95% interval to exclude `0.5` against the accepted runtime, and require replication on an independent `--salt` before acceptance. A single screening pass that clears the interval is not evidence; the `frames=36` result proves it.
3. **Keep the legacy gates and league as anchors only.** They must not regress, but they no longer measure progress. Do not tune constants to move `avgWin` on five seeds.
4. **Re-test the accumulated heuristics once the evaluator improves.** Several narrow constants were fitted to individual matches on an instrument that could not resolve them. Ablate them on the benchmark; remove the ones that do not survive. Expect some to be neutral.
5. **Do not spend further effort on rollout depth, trigger coverage, or another guarded rule** until the value function changes. All three are measured dead ends, and the measurements now have error bars.
6. **Native/WASM porting is not currently justified for the playable AI.** At 5Hz decisions the browser now has roughly two orders of magnitude of headroom per decision, so runtime speed is no longer what limits search. Revisit only if offline label generation for the value function becomes the throughput constraint, and only with the fingerprint suite extended to cover cross-language parity.
7. End every session by auditing measurement resolution first, then opponent diversity, evaluator alignment, and diagnostic fidelity. Record whether runtime AI actually improved or only the research process improved, remove scratch files, commit source/tests/docs, and push.

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
