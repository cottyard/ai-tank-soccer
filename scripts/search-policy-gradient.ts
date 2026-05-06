import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { loadWeightsPayload } from './coach-neural';
import {
  parsePolicyGradientArgs,
  runPolicyGradientCli,
  type PolicyGradientCliOptions
} from './train-policy-gradient';
import {
  compareRuntimeTraces,
  evaluateRuntimePolicy,
  traceRuntimePolicy,
  type RuntimeEvaluationOptions,
  type RuntimeEvaluationResult,
  type RuntimeTraceDelta,
  type RuntimeTraceSummary
} from '../src/ai/policyGate';
import type { NeuralWeights } from '../src/ai/neuralWeights';
import type { PolicyGradientTrainingResult } from '../src/ai/policyGradientTraining';

declare const process: {
  argv: string[];
  exitCode?: number;
};

export type PolicyGradientSearchEvaluation = RuntimeEvaluationResult;
export type PolicyGradientSearchTrace = RuntimeTraceSummary;

export type PolicyGradientSearchVariant = {
  trainingSeed: number;
  learningRate: number;
  epochs: number;
  ppoClip: number;
  temperature: number;
  startStateMode: PolicyGradientCliOptions['startStateMode'];
  openStartRatio?: number;
  advantageBaseline: PolicyGradientCliOptions['advantageBaseline'];
  runtimeWrapperMode: RuntimeWrapperSearchMode;
  runtimeTacticalRewriteWeight: number;
  opponentMode: PolicyGradientCliOptions['opponentMode'];
};

export type RuntimeWrapperSearchMode = 'none' | 'runtime-survivors-only' | 'tactical-downweight';

export type PolicyGradientSearchGate = {
  current: PolicyGradientSearchEvaluation;
  candidate: PolicyGradientSearchEvaluation;
  delta: PolicyGradientSearchEvaluation;
  trace?: PolicyGradientSearchTraceGate;
  seeds: PolicyGradientSearchSeedResult[];
};

export type PolicyGradientSearchTraceGate = {
  current: PolicyGradientSearchTrace;
  candidate: PolicyGradientSearchTrace;
  delta: RuntimeTraceDelta;
};

export type PolicyGradientSearchSeedResult = {
  seed: number;
  current: PolicyGradientSearchEvaluation;
  candidate: PolicyGradientSearchEvaluation;
  delta: PolicyGradientSearchEvaluation;
};

export type PolicyGradientSearchRow = {
  rank: number;
  variant: PolicyGradientSearchVariant;
  candidatePath: string;
  candidateMetricsPath: string;
  training: PolicyGradientCliOptions;
  standard: PolicyGradientSearchGate;
  holdout: PolicyGradientSearchGate;
};

export type PolicyGradientSearchOptions = {
  bestPath: string;
  outputDir: string;
  summaryPath: string;
  historyPath?: string;
  seed: number;
  matches: number;
  frames: number;
  batchSize: number;
  discount: number;
  startStateMode: string;
  advantageBaseline: string;
  actionMode: string;
  runtimeSurvivorsOnly: boolean;
  runtimeWrapperWeightMode: PolicyGradientCliOptions['runtimeWrapperWeightMode'];
  runtimeTacticalRewriteWeight: number;
  opponentMode: string;
  gateMatches: number;
  gateFrames: number;
  traceGate: boolean;
  standardSeeds: number[];
  holdoutSeeds: number[];
  training: PolicyGradientCliOptions;
  grid: {
    trainingSeeds: number[];
    learningRates: number[];
    epochs: number[];
    ppoClips: number[];
    temperatures: number[];
    startStateModes: PolicyGradientCliOptions['startStateMode'][];
    openStartRatios: Array<number | undefined>;
    advantageBaselines: PolicyGradientCliOptions['advantageBaseline'][];
    runtimeWrapperModes: RuntimeWrapperSearchMode[];
    runtimeTacticalRewriteWeights: number[];
    opponentModes: PolicyGradientCliOptions['opponentMode'][];
  };
  variants: PolicyGradientSearchVariant[];
};

export type PolicyGradientSearchResult = {
  bestPath: string;
  outputDir: string;
  summaryPath: string;
  historyPath?: string;
  seed: number;
  bestCandidatePath: string;
  best: PolicyGradientSearchRow;
  rows: PolicyGradientSearchRow[];
};

type PolicyGradientSearchDependencies = {
  train?: (options: PolicyGradientCliOptions) => PolicyGradientTrainingResult;
  evaluate?: (weights: NeuralWeights, options: RuntimeEvaluationOptions) => PolicyGradientSearchEvaluation;
  trace?: (
    weights: NeuralWeights,
    options: RuntimeEvaluationOptions & { seeds?: readonly number[] }
  ) => PolicyGradientSearchTrace;
};

const DEFAULT_SEED = 2026050212;
const DEFAULT_STANDARD_SEEDS = [19, 31, 43, 57, 71];
const DEFAULT_HOLDOUT_SEEDS = [83, 97, 109, 127, 149];
const DEFAULT_START_STATE_MODES: PolicyGradientCliOptions['startStateMode'][] = ['mixed'];
const DEFAULT_ADVANTAGE_BASELINES: PolicyGradientCliOptions['advantageBaseline'][] = ['start-team-time'];
const DEFAULT_OPPONENT_MODES: PolicyGradientCliOptions['opponentMode'][] = ['self'];

export function parsePolicyGradientSearchArgs(argv: readonly string[]): PolicyGradientSearchOptions {
  const seed = integerArg(argv, '--seed', DEFAULT_SEED);
  const startStateMode = startStateModeArg(argv, '--start-state-mode', DEFAULT_START_STATE_MODES[0]);
  const openStartRatio = optionalClamp01Arg(argv, '--open-start-ratio');
  const advantageBaseline = advantageBaselineArg(argv, '--advantage-baseline', DEFAULT_ADVANTAGE_BASELINES[0]);
  const opponentMode = opponentModeArg(argv, '--opponent-mode', DEFAULT_OPPONENT_MODES[0]);
  const runtimeSurvivorsOnly = argv.includes('--runtime-survivors-only');
  const runtimeWrapperWeightMode = runtimeWrapperWeightModeArg(argv, '--runtime-wrapper-weight-mode', 'none');
  const runtimeTacticalRewriteWeight = clamp01(numberArg(argv, '--runtime-tactical-rewrite-weight', 0.5));
  const grid = {
    trainingSeeds: seedListArg(argv, '--training-seeds', [seed]),
    learningRates: numberListArg(argv, '--learning-rates', [0.001, 0.0008, 0.0006]),
    epochs: integerListArg(argv, '--epochs-list', [1, 2]),
    ppoClips: numberListArg(argv, '--ppo-clips', [0.08, 0.12, 0.16]),
    temperatures: numberListArg(argv, '--temperatures', [1, 1.1]),
    startStateModes: startStateModeListArg(argv, '--start-state-modes', [startStateMode]),
    openStartRatios: optionalNumberListArg(argv, '--open-start-ratios', openStartRatio),
    advantageBaselines: advantageBaselineListArg(argv, '--advantage-baselines', [advantageBaseline]),
    runtimeWrapperModes: runtimeWrapperModeListArg(argv, '--runtime-wrapper-modes', [
      runtimeWrapperSearchMode(runtimeSurvivorsOnly, runtimeWrapperWeightMode)
    ]),
    runtimeTacticalRewriteWeights: numberListArg(argv, '--runtime-tactical-rewrite-weights', [runtimeTacticalRewriteWeight])
      .map(clamp01)
      .filter((value, index, values) => values.indexOf(value) === index),
    opponentModes: opponentModeListArg(argv, '--opponent-modes', [opponentMode])
  };
  const variants = expandVariants(grid);
  const matches = positiveIntegerArg(argv, '--matches', 240);
  const frames = positiveIntegerArg(argv, '--frames', 180);
  const batchSize = positiveIntegerArg(argv, '--batch-size', 192);
  const discount = numberArg(argv, '--discount', 0.996);
  const actionMode = stringArg(argv, '--action-mode') ?? 'runtime';
  const bestPath = stringArg(argv, '--best') ?? 'public/models/neural-best.json';
  const outputDir = stringArg(argv, '--output-dir') ?? `training-runs/policy-gradient-search-s${seed}`;

  return {
    bestPath,
    outputDir,
    summaryPath: stringArg(argv, '--summary-output') ?? join(outputDir, 'summary.json'),
    historyPath: stringArg(argv, '--history-output') ?? 'training-runs/policy-gradient-search-history.jsonl',
    seed,
    matches,
    frames,
    batchSize,
    discount,
    startStateMode,
    advantageBaseline,
    actionMode,
    runtimeSurvivorsOnly,
    runtimeWrapperWeightMode,
    runtimeTacticalRewriteWeight,
    opponentMode,
    gateMatches: positiveIntegerArg(argv, '--gate-matches', 2),
    gateFrames: positiveIntegerArg(argv, '--gate-frames', 360),
    traceGate: argv.includes('--trace-gate'),
    standardSeeds: seedListArg(argv, '--standard-seeds', DEFAULT_STANDARD_SEEDS),
    holdoutSeeds: seedListArg(argv, '--holdout-seeds', DEFAULT_HOLDOUT_SEEDS),
    training: parseTrainingOptions(argv, seed, {
      bestPath,
      matches,
      frames,
      batchSize,
      discount,
      startStateMode,
      advantageBaseline,
      actionMode,
      runtimeSurvivorsOnly,
      runtimeWrapperWeightMode,
      runtimeTacticalRewriteWeight,
      opponentMode
    }),
    grid,
    variants
  };
}

export function runPolicyGradientSearch(
  options: PolicyGradientSearchOptions,
  dependencies: PolicyGradientSearchDependencies = {}
): PolicyGradientSearchResult {
  const train = dependencies.train ?? runPolicyGradientCli;
  const evaluate = dependencies.evaluate ?? evaluateRuntimePolicy;
  const trace = dependencies.trace ?? traceRuntimePolicy;
  const currentWeights = loadWeightsPayload(readFileSync(options.bestPath, 'utf8'));
  ensureDirectory(options.outputDir);

  const rows = options.variants.map((variant, index) => {
    const variantId = [
      `v${String(index + 1).padStart(2, '0')}`,
      `seed${variant.trainingSeed}`,
      `lr${slugNumber(variant.learningRate)}`,
      `e${variant.epochs}`,
      `clip${slugNumber(variant.ppoClip)}`,
      `t${slugNumber(variant.temperature)}`,
      `start${slugText(variant.startStateMode)}`,
      variant.openStartRatio === undefined ? undefined : `open${slugNumber(variant.openStartRatio)}`,
      `baseline${slugText(variant.advantageBaseline)}`,
      `opp${slugText(variant.opponentMode)}`,
      options.grid.runtimeWrapperModes.length > 1 || options.grid.runtimeTacticalRewriteWeights.length > 1
        ? `wrap${runtimeWrapperModeSlug(variant.runtimeWrapperMode)}`
        : undefined,
      variant.runtimeWrapperMode === 'tactical-downweight' && options.grid.runtimeTacticalRewriteWeights.length > 1
        ? `tacw${slugNumber(variant.runtimeTacticalRewriteWeight)}`
        : undefined
    ].filter((part): part is string => part !== undefined).join('-');
    const candidatePath = join(options.outputDir, `${variantId}.json`);
    const candidateMetricsPath = join(options.outputDir, `${variantId}-metrics.json`);
    const training: PolicyGradientCliOptions = {
      ...options.training,
      seed: variant.trainingSeed,
      input: options.bestPath,
      output: candidatePath,
      metricsOutput: candidateMetricsPath,
      learningRate: variant.learningRate,
      epochs: variant.epochs,
      ppoClip: variant.ppoClip,
      temperature: variant.temperature,
      startStateMode: variant.startStateMode,
      openStartRatio: variant.openStartRatio,
      advantageBaseline: variant.advantageBaseline,
      runtimeSurvivorsOnly: variant.runtimeWrapperMode === 'runtime-survivors-only',
      runtimeWrapperWeightMode: variant.runtimeWrapperMode === 'tactical-downweight' ? 'tactical-downweight' : 'none',
      runtimeTacticalRewriteWeight: variant.runtimeTacticalRewriteWeight,
      opponentMode: variant.opponentMode
    };

    const trained = train(training);
    const candidateWeights = existsSync(candidatePath)
      ? loadWeightsPayload(readFileSync(candidatePath, 'utf8'))
      : [...trained.weights];

    return {
      rank: 0,
      variant,
      candidatePath,
      candidateMetricsPath,
      training,
      standard: evaluateGate(currentWeights, candidateWeights, options.standardSeeds, options, evaluate, trace),
      holdout: evaluateGate(currentWeights, candidateWeights, options.holdoutSeeds, options, evaluate, trace)
    };
  }).sort((a, b) => compareRows(a, b))
    .map((row, index) => ({ ...row, rank: index + 1 }));

  const best = rows[0];
  if (!best) {
    throw new Error('Policy-gradient search requires at least one variant');
  }

  const result = {
    bestPath: options.bestPath,
    outputDir: options.outputDir,
    summaryPath: options.summaryPath,
    historyPath: options.historyPath,
    seed: options.seed,
    bestCandidatePath: best.candidatePath,
    best,
    rows
  };
  writeSummary(options.summaryPath, result);
  if (options.historyPath) {
    appendHistory(options.historyPath, result);
  }
  return result;
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const result = runPolicyGradientSearch(parsePolicyGradientSearchArgs(argv));
    console.log(`variants=${result.rows.length}`);
    console.log(`bestRank=${result.best.rank}`);
    console.log(`bestTrainingSeed=${result.best.variant.trainingSeed}`);
    console.log(`bestScoreDelta=${result.best.standard.delta.score.toFixed(3)}`);
    console.log(`bestHoldoutScoreDelta=${result.best.holdout.delta.score.toFixed(3)}`);
    if (result.best.holdout.trace) {
      console.log(`bestHoldoutFinalActionChangeRate=${result.best.holdout.trace.delta.finalActionDistributionChangeRate.toFixed(3)}`);
    }
    console.log(`bestGoals=${result.best.standard.candidate.goalsFor}-${result.best.standard.candidate.goalsAgainst}`);
    console.log(`bestHoldoutGoals=${result.best.holdout.candidate.goalsFor}-${result.best.holdout.candidate.goalsAgainst}`);
    console.log(`bestLearningRate=${result.best.variant.learningRate}`);
    console.log(`bestEpochs=${result.best.variant.epochs}`);
    console.log(`bestPpoClip=${result.best.variant.ppoClip}`);
    console.log(`bestTemperature=${result.best.variant.temperature}`);
    console.log(`bestStartStateMode=${result.best.variant.startStateMode}`);
    console.log(`bestAdvantageBaseline=${result.best.variant.advantageBaseline}`);
    console.log(`bestOpponentMode=${result.best.variant.opponentMode}`);
    console.log(`bestCandidate=${result.bestCandidatePath}`);
    console.log(`summaryOut=${result.summaryPath}`);
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

function parseTrainingOptions(
  argv: readonly string[],
  seed: number,
  base: {
    bestPath: string;
    matches: number;
    frames: number;
    batchSize: number;
    discount: number;
    startStateMode: string;
    advantageBaseline: string;
    actionMode: string;
    runtimeSurvivorsOnly: boolean;
    runtimeWrapperWeightMode: PolicyGradientCliOptions['runtimeWrapperWeightMode'];
    runtimeTacticalRewriteWeight: number;
    opponentMode: string;
  }
): PolicyGradientCliOptions {
  const trainingArgs = [
    '--native',
    '--input',
    base.bestPath,
    '--seed',
    String(seed),
    '--matches',
    String(base.matches),
    '--frames',
    String(base.frames),
    '--epochs',
    '1',
    '--batch-size',
    String(base.batchSize),
    '--learning-rate',
    '0.001',
    '--ppo-clip',
    '0.12',
    '--temperature',
    '1.1',
    '--discount',
    String(base.discount),
    '--start-state-mode',
    base.startStateMode,
    '--advantage-baseline',
    base.advantageBaseline,
    '--action-mode',
    base.actionMode,
    ...(base.runtimeSurvivorsOnly ? ['--runtime-survivors-only'] : []),
    '--runtime-wrapper-weight-mode',
    base.runtimeWrapperWeightMode,
    '--runtime-tactical-rewrite-weight',
    String(base.runtimeTacticalRewriteWeight),
    '--opponent-mode',
    base.opponentMode,
    '--league-current-weight',
    stringArg(argv, '--league-current-weight') ?? '1',
    '--league-traditional-weight',
    stringArg(argv, '--league-traditional-weight') ?? '0.15'
  ];
  const nativeBin = stringArg(argv, '--native-bin');

  for (const path of stringArgs(argv, '--league-opponent-weights')) {
    trainingArgs.push('--league-opponent-weights', path);
  }
  if (nativeBin) {
    trainingArgs.push('--native-bin', nativeBin);
  }
  if (argv.includes('--js-trainer')) {
    const nativeIndex = trainingArgs.indexOf('--native');
    trainingArgs.splice(nativeIndex, 1);
  }

  return parsePolicyGradientArgs(trainingArgs);
}

function evaluateGate(
  currentWeights: NeuralWeights,
  candidateWeights: NeuralWeights,
  seeds: readonly number[],
  options: PolicyGradientSearchOptions,
  evaluate: (weights: NeuralWeights, options: RuntimeEvaluationOptions) => PolicyGradientSearchEvaluation,
  trace?: (weights: NeuralWeights, options: RuntimeEvaluationOptions & { seeds?: readonly number[] }) => PolicyGradientSearchTrace
): PolicyGradientSearchGate {
  const seedResults = seeds.map((seed) => {
    const current = evaluate(currentWeights, {
      seed,
      matches: options.gateMatches,
      frames: options.gateFrames
    });
    const candidate = evaluate(candidateWeights, {
      seed,
      matches: options.gateMatches,
      frames: options.gateFrames
    });
    return {
      seed,
      current,
      candidate,
      delta: deltaEvaluation(candidate, current)
    };
  });

  const current = aggregateEvaluations(seedResults.map((seed) => seed.current));
  const candidate = aggregateEvaluations(seedResults.map((seed) => seed.candidate));
  return {
    current,
    candidate,
    delta: deltaEvaluation(candidate, current),
    trace: options.traceGate ? evaluateTraceGate(currentWeights, candidateWeights, seeds, options, trace) : undefined,
    seeds: seedResults
  };
}

function evaluateTraceGate(
  currentWeights: NeuralWeights,
  candidateWeights: NeuralWeights,
  seeds: readonly number[],
  options: PolicyGradientSearchOptions,
  trace: ((weights: NeuralWeights, options: RuntimeEvaluationOptions & { seeds?: readonly number[] }) => PolicyGradientSearchTrace) | undefined
): PolicyGradientSearchTraceGate {
  const traceRuntime = trace ?? (() => {
    throw new Error('Trace gate requires a trace dependency or a runtime trace implementation');
  });
  const traceOptions = {
    seeds,
    matches: options.gateMatches,
    frames: options.gateFrames
  };
  const current = traceRuntime(currentWeights, traceOptions);
  const candidate = traceRuntime(candidateWeights, traceOptions);
  return {
    current,
    candidate,
    delta: compareRuntimeTraces(candidate, current)
  };
}

function aggregateEvaluations(evaluations: readonly PolicyGradientSearchEvaluation[]): PolicyGradientSearchEvaluation {
  if (evaluations.length === 0) {
    return emptyEvaluation();
  }

  const total = evaluations.reduce((sum, row) => ({
    score: sum.score + row.score,
    goalDiff: sum.goalDiff + row.goalDiff,
    ballProgress: sum.ballProgress + row.ballProgress,
    goalsFor: sum.goalsFor + row.goalsFor,
    goalsAgainst: sum.goalsAgainst + row.goalsAgainst,
    winProxy: sum.winProxy + row.winProxy
  }), emptyEvaluation());

  return {
    score: total.score / evaluations.length,
    goalDiff: total.goalsFor - total.goalsAgainst,
    ballProgress: total.ballProgress / evaluations.length,
    goalsFor: total.goalsFor,
    goalsAgainst: total.goalsAgainst,
    winProxy: total.winProxy / evaluations.length
  };
}

function emptyEvaluation(): PolicyGradientSearchEvaluation {
  return {
    score: 0,
    goalDiff: 0,
    ballProgress: 0,
    goalsFor: 0,
    goalsAgainst: 0,
    winProxy: 0
  };
}

function deltaEvaluation(
  candidate: PolicyGradientSearchEvaluation,
  current: PolicyGradientSearchEvaluation
): PolicyGradientSearchEvaluation {
  return {
    score: candidate.score - current.score,
    goalDiff: candidate.goalDiff - current.goalDiff,
    ballProgress: candidate.ballProgress - current.ballProgress,
    goalsFor: candidate.goalsFor - current.goalsFor,
    goalsAgainst: candidate.goalsAgainst - current.goalsAgainst,
    winProxy: candidate.winProxy - current.winProxy
  };
}

function compareRows(a: PolicyGradientSearchRow, b: PolicyGradientSearchRow): number {
  return promotionSafeRank(b) - promotionSafeRank(a) ||
    standardTraceSafetyRank(b) - standardTraceSafetyRank(a) ||
    b.holdout.delta.score - a.holdout.delta.score ||
    b.holdout.delta.goalDiff - a.holdout.delta.goalDiff ||
    b.holdout.delta.winProxy - a.holdout.delta.winProxy ||
    b.holdout.delta.ballProgress - a.holdout.delta.ballProgress ||
    traceVisibilityRank(b.holdout) - traceVisibilityRank(a.holdout) ||
    traceVisibilityRank(b.standard) - traceVisibilityRank(a.standard) ||
    b.standard.delta.score - a.standard.delta.score ||
    b.standard.delta.goalDiff - a.standard.delta.goalDiff ||
    b.standard.delta.winProxy - a.standard.delta.winProxy ||
    b.standard.delta.ballProgress - a.standard.delta.ballProgress;
}

function traceVisibilityRank(gate: PolicyGradientSearchGate): number {
  return gate.trace?.delta.finalActionDistributionChangeRate ?? 0;
}

function standardTraceSafetyRank(row: PolicyGradientSearchRow): number {
  const delta = row.standard.trace?.delta;
  if (!delta) {
    return 1;
  }
  return delta.staminaConserveRate <= 0 && delta.criticalStaminaRegulationRate <= 0 ? 1 : 0;
}

function promotionSafeRank(row: PolicyGradientSearchRow): number {
  return row.standard.delta.score >= 0 &&
    row.standard.delta.goalDiff >= 0 &&
    row.standard.delta.winProxy >= 0
    ? 1
    : 0;
}

function writeSummary(path: string, result: PolicyGradientSearchResult): void {
  ensureParentDirectory(path);
  writeFileSync(path, `${JSON.stringify({
    generatedAt: new Date().toISOString(),
    bestPath: result.bestPath,
    outputDir: result.outputDir,
    bestCandidatePath: result.bestCandidatePath,
    best: result.best,
    rows: result.rows
  }, null, 2)}\n`, 'utf8');
}

function appendHistory(path: string, result: PolicyGradientSearchResult): void {
  ensureParentDirectory(path);
  appendFileSync(path, `${JSON.stringify({
    generatedAt: new Date().toISOString(),
    seed: result.seed,
    variants: result.rows.length,
    bestCandidatePath: result.bestCandidatePath,
    bestTrainingSeed: result.best.variant.trainingSeed,
    bestStandardScoreDelta: result.best.standard.delta.score,
    bestStandardGoalDelta: result.best.standard.delta.goalDiff,
    bestStandardWinDelta: result.best.standard.delta.winProxy,
    bestHoldoutScoreDelta: result.best.holdout.delta.score,
    bestHoldoutGoalDelta: result.best.holdout.delta.goalDiff,
    bestHoldoutWinDelta: result.best.holdout.delta.winProxy,
    bestStandardFinalActionChangeRate: result.best.standard.trace?.delta.finalActionDistributionChangeRate,
    bestHoldoutFinalActionChangeRate: result.best.holdout.trace?.delta.finalActionDistributionChangeRate,
    bestLearningRate: result.best.variant.learningRate,
    bestEpochs: result.best.variant.epochs,
    bestPpoClip: result.best.variant.ppoClip,
    bestTemperature: result.best.variant.temperature,
    bestStartStateMode: result.best.variant.startStateMode,
    bestOpenStartRatio: result.best.variant.openStartRatio,
    bestAdvantageBaseline: result.best.variant.advantageBaseline,
    bestRuntimeWrapperMode: result.best.variant.runtimeWrapperMode,
    runtimeSurvivorsOnly: result.best.training.runtimeSurvivorsOnly,
    bestOpponentMode: result.best.variant.opponentMode,
    runtimeWrapperWeightMode: result.best.training.runtimeWrapperWeightMode,
    bestRuntimeTacticalRewriteWeight: result.best.variant.runtimeTacticalRewriteWeight
  })}\n`, 'utf8');
}

function expandVariants(grid: {
  trainingSeeds: readonly number[];
  learningRates: readonly number[];
  epochs: readonly number[];
  ppoClips: readonly number[];
  temperatures: readonly number[];
  startStateModes: readonly PolicyGradientCliOptions['startStateMode'][];
  openStartRatios: readonly (number | undefined)[];
  advantageBaselines: readonly PolicyGradientCliOptions['advantageBaseline'][];
  runtimeWrapperModes: readonly RuntimeWrapperSearchMode[];
  runtimeTacticalRewriteWeights: readonly number[];
  opponentModes: readonly PolicyGradientCliOptions['opponentMode'][];
}): PolicyGradientSearchVariant[] {
  const variants: PolicyGradientSearchVariant[] = [];
  for (const trainingSeed of grid.trainingSeeds) {
    for (const learningRate of grid.learningRates) {
      for (const epochs of grid.epochs) {
        for (const ppoClip of grid.ppoClips) {
          for (const temperature of grid.temperatures) {
            for (const startStateMode of grid.startStateModes) {
              const openStartRatios = startStateMode === 'mixed' ? grid.openStartRatios : [undefined];
              for (const openStartRatio of openStartRatios) {
                for (const advantageBaseline of grid.advantageBaselines) {
                  for (const runtimeWrapperMode of grid.runtimeWrapperModes) {
                    const rewriteWeights = runtimeWrapperMode === 'tactical-downweight'
                      ? grid.runtimeTacticalRewriteWeights
                      : [0.5];
                    for (const runtimeTacticalRewriteWeight of rewriteWeights) {
                      for (const opponentMode of grid.opponentModes) {
                        variants.push({
                          trainingSeed,
                          learningRate,
                          epochs,
                          ppoClip,
                          temperature,
                          startStateMode,
                          openStartRatio,
                          advantageBaseline,
                          runtimeWrapperMode,
                          runtimeTacticalRewriteWeight,
                          opponentMode
                        });
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return variants;
}

function ensureDirectory(path: string): void {
  if (!existsSync(path)) {
    mkdirSync(path, { recursive: true });
  }
}

function ensureParentDirectory(path: string): void {
  const parent = dirname(path);
  if (parent && parent !== '.' && !existsSync(parent)) {
    mkdirSync(parent, { recursive: true });
  }
}

function slugNumber(value: number): string {
  return String(value).replace('-', 'm').replace('.', 'p');
}

function slugText(value: string): string {
  return value.replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '');
}

function runtimeWrapperModeSlug(mode: RuntimeWrapperSearchMode): string {
  return mode === 'runtime-survivors-only' ? 'survivors' : slugText(mode);
}

function numberListArg(argv: readonly string[], name: string, fallback: readonly number[]): number[] {
  const value = listArgValue(argv, name);
  if (!value) {
    return [...fallback];
  }
  const values = value.split(',')
    .map((part) => Number(part.trim()))
    .filter((part) => Number.isFinite(part));
  return values.length > 0 ? values : [...fallback];
}

function optionalNumberListArg(
  argv: readonly string[],
  name: string,
  singleFallback?: number
): Array<number | undefined> {
  const value = listArgValue(argv, name);
  if (!value) {
    return singleFallback === undefined ? [undefined] : [singleFallback];
  }
  const values = value.split(',')
    .map((part) => Number(part.trim()))
    .filter((part) => Number.isFinite(part))
    .map(clamp01)
    .filter((part, index, parts) => parts.indexOf(part) === index);
  return values.length > 0 ? values : singleFallback === undefined ? [undefined] : [singleFallback];
}

function integerListArg(argv: readonly string[], name: string, fallback: readonly number[]): number[] {
  return numberListArg(argv, name, fallback)
    .map((value) => Math.max(0, Math.floor(value)))
    .filter((value, index, values) => value > 0 && values.indexOf(value) === index);
}

function startStateModeArg(
  argv: readonly string[],
  name: string,
  fallback: PolicyGradientCliOptions['startStateMode']
): PolicyGradientCliOptions['startStateMode'] {
  return firstValid(startStateModeListArg(argv, name, [fallback]), fallback);
}

function advantageBaselineArg(
  argv: readonly string[],
  name: string,
  fallback: PolicyGradientCliOptions['advantageBaseline']
): PolicyGradientCliOptions['advantageBaseline'] {
  return firstValid(advantageBaselineListArg(argv, name, [fallback]), fallback);
}

function opponentModeArg(
  argv: readonly string[],
  name: string,
  fallback: PolicyGradientCliOptions['opponentMode']
): PolicyGradientCliOptions['opponentMode'] {
  return firstValid(opponentModeListArg(argv, name, [fallback]), fallback);
}

function runtimeWrapperWeightModeArg(
  argv: readonly string[],
  name: string,
  fallback: PolicyGradientCliOptions['runtimeWrapperWeightMode']
): PolicyGradientCliOptions['runtimeWrapperWeightMode'] {
  const value = stringArg(argv, name);
  return value === 'none' || value === 'tactical-downweight'
    ? value
    : fallback;
}

function startStateModeListArg(
  argv: readonly string[],
  name: string,
  fallback: readonly PolicyGradientCliOptions['startStateMode'][]
): PolicyGradientCliOptions['startStateMode'][] {
  return enumListArg(argv, name, fallback, isStartStateMode);
}

function advantageBaselineListArg(
  argv: readonly string[],
  name: string,
  fallback: readonly PolicyGradientCliOptions['advantageBaseline'][]
): PolicyGradientCliOptions['advantageBaseline'][] {
  return enumListArg(argv, name, fallback, isAdvantageBaseline);
}

function opponentModeListArg(
  argv: readonly string[],
  name: string,
  fallback: readonly PolicyGradientCliOptions['opponentMode'][]
): PolicyGradientCliOptions['opponentMode'][] {
  return enumListArg(argv, name, fallback, isOpponentMode);
}

function runtimeWrapperModeListArg(
  argv: readonly string[],
  name: string,
  fallback: readonly RuntimeWrapperSearchMode[]
): RuntimeWrapperSearchMode[] {
  return enumListArg(argv, name, fallback, isRuntimeWrapperSearchMode);
}

function enumListArg<T extends string>(
  argv: readonly string[],
  name: string,
  fallback: readonly T[],
  isValid: (value: string) => value is T
): T[] {
  const value = listArgValue(argv, name);
  if (!value) {
    return [...fallback];
  }
  const values = value.split(',')
    .map((part) => part.trim())
    .filter(isValid)
    .filter((part, index, parts) => parts.indexOf(part) === index);
  return values.length > 0 ? values : [...fallback];
}

function firstValid<T extends string>(values: readonly T[], fallback: T): T {
  return values[0] ?? fallback;
}

function isStartStateMode(value: string): value is PolicyGradientCliOptions['startStateMode'] {
  return value === 'open' ||
    value === 'outcome-curriculum' ||
    value === 'own-goal-defense' ||
    value === 'corner-fight' ||
    value === 'loose-ball-contest' ||
    value === 'mixed';
}

function isAdvantageBaseline(value: string): value is PolicyGradientCliOptions['advantageBaseline'] {
  return value === 'global' || value === 'start-team-time' || value === 'learned';
}

function isOpponentMode(value: string): value is PolicyGradientCliOptions['opponentMode'] {
  return value === 'self' || value === 'traditional' || value === 'league';
}

function isRuntimeWrapperSearchMode(value: string): value is RuntimeWrapperSearchMode {
  return value === 'none' || value === 'runtime-survivors-only' || value === 'tactical-downweight';
}

function runtimeWrapperSearchMode(
  runtimeSurvivorsOnly: boolean,
  runtimeWrapperWeightMode: PolicyGradientCliOptions['runtimeWrapperWeightMode']
): RuntimeWrapperSearchMode {
  if (runtimeSurvivorsOnly) {
    return 'runtime-survivors-only';
  }
  return runtimeWrapperWeightMode === 'tactical-downweight' ? 'tactical-downweight' : 'none';
}

function seedListArg(argv: readonly string[], name: string, fallback: readonly number[]): number[] {
  const value = listArgValue(argv, name);
  if (!value) {
    return [...fallback];
  }
  const seeds = value.split(',')
    .map((part) => Math.floor(Number(part.trim())))
    .filter((seed) => Number.isFinite(seed));
  return seeds.length > 0 ? seeds : [...fallback];
}

function stringArg(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  return index === -1 || index === argv.length - 1 ? undefined : argv[index + 1];
}

function listArgValue(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index === -1 || index === argv.length - 1) {
    return undefined;
  }
  const values: string[] = [];
  for (let cursor = index + 1; cursor < argv.length; cursor += 1) {
    const value = argv[cursor];
    if (value.startsWith('--')) {
      break;
    }
    values.push(value);
  }
  return values.length > 0 ? values.join(',') : undefined;
}

function stringArgs(argv: readonly string[], name: string): string[] {
  const values: string[] = [];
  for (let index = 0; index < argv.length - 1; index += 1) {
    if (argv[index] === name) {
      values.push(argv[index + 1]);
    }
  }
  return values;
}

function numberArg(argv: readonly string[], name: string, fallback: number): number {
  const value = stringArg(argv, name);
  if (value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function optionalClamp01Arg(argv: readonly string[], name: string): number | undefined {
  const value = stringArg(argv, name);
  if (value === undefined) {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? clamp01(parsed) : undefined;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function integerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.floor(numberArg(argv, name, fallback));
}

function positiveIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.max(1, integerArg(argv, name, fallback));
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/search-policy-gradient.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/search-policy-gradient.js')) {
  main();
}
