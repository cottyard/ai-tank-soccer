import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { loadWeightsPayload } from './coach-neural';
import {
  parsePolicyGradientArgs,
  runPolicyGradientCli,
  type PolicyGradientCliOptions
} from './train-policy-gradient';
import {
  evaluateRuntimePolicy,
  type RuntimeEvaluationOptions,
  type RuntimeEvaluationResult
} from '../src/ai/policyGate';
import type { NeuralWeights } from '../src/ai/neuralWeights';
import type { PolicyGradientTrainingResult } from '../src/ai/policyGradientTraining';

declare const process: {
  argv: string[];
  exitCode?: number;
};

export type PolicyGradientSearchEvaluation = RuntimeEvaluationResult;

export type PolicyGradientSearchVariant = {
  trainingSeed: number;
  learningRate: number;
  epochs: number;
  ppoClip: number;
  temperature: number;
  startStateMode: PolicyGradientCliOptions['startStateMode'];
  advantageBaseline: PolicyGradientCliOptions['advantageBaseline'];
  opponentMode: PolicyGradientCliOptions['opponentMode'];
};

export type PolicyGradientSearchGate = {
  current: PolicyGradientSearchEvaluation;
  candidate: PolicyGradientSearchEvaluation;
  delta: PolicyGradientSearchEvaluation;
  seeds: PolicyGradientSearchSeedResult[];
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
  opponentMode: string;
  gateMatches: number;
  gateFrames: number;
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
    advantageBaselines: PolicyGradientCliOptions['advantageBaseline'][];
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
  const advantageBaseline = advantageBaselineArg(argv, '--advantage-baseline', DEFAULT_ADVANTAGE_BASELINES[0]);
  const opponentMode = opponentModeArg(argv, '--opponent-mode', DEFAULT_OPPONENT_MODES[0]);
  const grid = {
    trainingSeeds: seedListArg(argv, '--training-seeds', [seed]),
    learningRates: numberListArg(argv, '--learning-rates', [0.001, 0.0008, 0.0006]),
    epochs: integerListArg(argv, '--epochs-list', [1, 2]),
    ppoClips: numberListArg(argv, '--ppo-clips', [0.08, 0.12, 0.16]),
    temperatures: numberListArg(argv, '--temperatures', [1, 1.1]),
    startStateModes: startStateModeListArg(argv, '--start-state-modes', [startStateMode]),
    advantageBaselines: advantageBaselineListArg(argv, '--advantage-baselines', [advantageBaseline]),
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
    opponentMode,
    gateMatches: positiveIntegerArg(argv, '--gate-matches', 2),
    gateFrames: positiveIntegerArg(argv, '--gate-frames', 360),
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
      `baseline${slugText(variant.advantageBaseline)}`,
      `opp${slugText(variant.opponentMode)}`
    ].join('-');
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
      advantageBaseline: variant.advantageBaseline,
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
      standard: evaluateGate(currentWeights, candidateWeights, options.standardSeeds, options, evaluate),
      holdout: evaluateGate(currentWeights, candidateWeights, options.holdoutSeeds, options, evaluate)
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
  evaluate: (weights: NeuralWeights, options: RuntimeEvaluationOptions) => PolicyGradientSearchEvaluation
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
    seeds: seedResults
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
    b.holdout.delta.score - a.holdout.delta.score ||
    b.holdout.delta.goalDiff - a.holdout.delta.goalDiff ||
    b.holdout.delta.winProxy - a.holdout.delta.winProxy ||
    b.holdout.delta.ballProgress - a.holdout.delta.ballProgress ||
    b.standard.delta.score - a.standard.delta.score ||
    b.standard.delta.goalDiff - a.standard.delta.goalDiff ||
    b.standard.delta.winProxy - a.standard.delta.winProxy ||
    b.standard.delta.ballProgress - a.standard.delta.ballProgress;
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
    bestLearningRate: result.best.variant.learningRate,
    bestEpochs: result.best.variant.epochs,
    bestPpoClip: result.best.variant.ppoClip,
    bestTemperature: result.best.variant.temperature,
    bestStartStateMode: result.best.variant.startStateMode,
    bestAdvantageBaseline: result.best.variant.advantageBaseline,
    bestOpponentMode: result.best.variant.opponentMode
  })}\n`, 'utf8');
}

function expandVariants(grid: {
  trainingSeeds: readonly number[];
  learningRates: readonly number[];
  epochs: readonly number[];
  ppoClips: readonly number[];
  temperatures: readonly number[];
  startStateModes: readonly PolicyGradientCliOptions['startStateMode'][];
  advantageBaselines: readonly PolicyGradientCliOptions['advantageBaseline'][];
  opponentModes: readonly PolicyGradientCliOptions['opponentMode'][];
}): PolicyGradientSearchVariant[] {
  const variants: PolicyGradientSearchVariant[] = [];
  for (const trainingSeed of grid.trainingSeeds) {
    for (const learningRate of grid.learningRates) {
      for (const epochs of grid.epochs) {
        for (const ppoClip of grid.ppoClips) {
          for (const temperature of grid.temperatures) {
            for (const startStateMode of grid.startStateModes) {
              for (const advantageBaseline of grid.advantageBaselines) {
                for (const opponentMode of grid.opponentModes) {
                  variants.push({
                    trainingSeed,
                    learningRate,
                    epochs,
                    ppoClip,
                    temperature,
                    startStateMode,
                    advantageBaseline,
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

function numberListArg(argv: readonly string[], name: string, fallback: readonly number[]): number[] {
  const value = stringArg(argv, name);
  if (!value) {
    return [...fallback];
  }
  const values = value.split(',')
    .map((part) => Number(part.trim()))
    .filter((part) => Number.isFinite(part));
  return values.length > 0 ? values : [...fallback];
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

function enumListArg<T extends string>(
  argv: readonly string[],
  name: string,
  fallback: readonly T[],
  isValid: (value: string) => value is T
): T[] {
  const value = stringArg(argv, name);
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

function seedListArg(argv: readonly string[], name: string, fallback: readonly number[]): number[] {
  const value = stringArg(argv, name);
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
