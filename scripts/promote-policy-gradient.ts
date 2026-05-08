import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import {
  loadWeightsPayload,
  serializeWeightsPayload
} from './coach-neural';
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

export type PromotionLoopEvaluation = RuntimeEvaluationResult;

export type PromotionLoopGate = {
  name: 'standard' | 'holdout';
  accepted: boolean;
  reason?: string;
  current: PromotionLoopEvaluation;
  candidate: PromotionLoopEvaluation;
  delta: PromotionLoopEvaluation;
  seeds: PromotionLoopSeedResult[];
};

export type PromotionLoopSeedResult = {
  seed: number;
  current: PromotionLoopEvaluation;
  candidate: PromotionLoopEvaluation;
  delta: PromotionLoopEvaluation;
};

export type PromotionLoopOptions = {
  bestPath: string;
  candidateInputPath?: string;
  candidatePath: string;
  candidateMetricsPath?: string;
  summaryPath: string;
  historyPath?: string;
  iterations: number;
  promote: boolean;
  seed: number;
  gateMatches: number;
  gateFrames: number;
  standardSeeds: number[];
  holdoutSeeds: number[];
  minScoreDelta: number;
  maxGoalDiffRegression: number;
  maxWinProxyRegression: number;
  training: PolicyGradientCliOptions;
};

export type PromotionLoopResult = {
  promoted: boolean;
  rejectionReason?: string;
  bestPath: string;
  candidatePath: string;
  candidateMetricsPath?: string;
  summaryPath: string;
  historyPath?: string;
  standard: PromotionLoopGate;
  holdout: PromotionLoopGate;
  training: PolicyGradientCliOptions;
};

type PromotionLoopDependencies = {
  train?: (options: PolicyGradientCliOptions) => PolicyGradientTrainingResult;
  evaluate?: (weights: NeuralWeights, options: RuntimeEvaluationOptions) => PromotionLoopEvaluation;
  beforePromote?: (result: PromotionLoopResult) => void;
};

const DEFAULT_SEED = 2026050208;
const DEFAULT_STANDARD_SEEDS = [19, 31, 43, 57, 71];
const DEFAULT_HOLDOUT_SEEDS = [83, 97, 109, 127, 149];

export function parsePromotionLoopArgs(argv: readonly string[]): PromotionLoopOptions {
  const seed = integerArg(argv, '--seed', DEFAULT_SEED);
  const candidateInputPath = stringArg(argv, '--candidate-input');
  return {
    bestPath: stringArg(argv, '--best') ?? 'public/models/neural-best.json',
    candidateInputPath,
    candidatePath: candidateInputPath ?? stringArg(argv, '--candidate-output') ??
      `training-runs/neural-pg-promotion-candidate-s${seed}.json`,
    candidateMetricsPath: candidateInputPath
      ? stringArg(argv, '--candidate-metrics-output')
      : stringArg(argv, '--candidate-metrics-output') ??
      `training-runs/neural-pg-promotion-candidate-metrics-s${seed}.json`,
    summaryPath: stringArg(argv, '--summary-output') ??
      `training-runs/neural-promotion-summary-s${seed}.json`,
    historyPath: stringArg(argv, '--history-output') ?? 'training-runs/neural-promotion-history.jsonl',
    iterations: positiveIntegerArg(argv, '--iterations', 1),
    promote: !argv.includes('--no-promote'),
    seed,
    gateMatches: positiveIntegerArg(argv, '--gate-matches', 4),
    gateFrames: positiveIntegerArg(argv, '--gate-frames', 600),
    standardSeeds: seedListArg(argv, '--standard-seeds', DEFAULT_STANDARD_SEEDS),
    holdoutSeeds: seedListArg(argv, '--holdout-seeds', DEFAULT_HOLDOUT_SEEDS),
    minScoreDelta: numberArg(argv, '--min-score-delta', 0),
    maxGoalDiffRegression: numberArg(argv, '--max-goal-diff-regression', 0),
    maxWinProxyRegression: numberArg(argv, '--max-win-proxy-regression', 0.025),
    training: parseTrainingOptions(argv, seed)
  };
}

export function runPromotionLoop(
  options: PromotionLoopOptions,
  dependencies: PromotionLoopDependencies = {}
): PromotionLoopResult {
  const train = dependencies.train ?? runPolicyGradientCli;
  const evaluate = dependencies.evaluate ?? evaluateRuntimePolicy;
  let currentWeights = loadWeightsPayload(readFileSync(options.bestPath, 'utf8'));
  let result: PromotionLoopResult | undefined;

  for (let iteration = 1; iteration <= options.iterations; iteration += 1) {
    const candidatePath = iterationPath(options.candidatePath, iteration, options.iterations);
    const candidateMetricsPath = options.candidateMetricsPath
      ? iterationPath(options.candidateMetricsPath, iteration, options.iterations)
      : undefined;
    if (!options.candidateInputPath) {
      ensureParentDirectory(candidatePath);
    }
    if (!options.candidateInputPath && candidateMetricsPath) {
      ensureParentDirectory(candidateMetricsPath);
    }

    const training = {
      ...options.training,
      seed: options.seed + iteration - 1,
      input: options.bestPath,
      output: candidatePath,
      metricsOutput: candidateMetricsPath
    };
    const trained = options.candidateInputPath ? undefined : train(training);
    const candidatePayload = existsSync(candidatePath)
      ? readFileSync(candidatePath, 'utf8')
      : undefined;
    const candidateWeights = candidatePayload
      ? loadWeightsPayload(candidatePayload)
      : trained
        ? [...trained.weights]
        : loadWeightsPayload(readFileSync(requiredCandidateInputPath(options), 'utf8'));
    const resultTraining = options.candidateInputPath && candidatePayload
      ? trainingFromCandidateMetadata(training, candidatePayload)
      : training;
    const standard = evaluateGate('standard', currentWeights, candidateWeights, options.standardSeeds, options, evaluate);
    const holdout = standard.accepted
      ? evaluateGate('holdout', currentWeights, candidateWeights, options.holdoutSeeds, options, evaluate)
      : skippedGate('holdout', standard.reason ?? 'standard gate failed');
    const promoted = options.promote && standard.accepted && holdout.accepted;
    const rejectionReason = promoted
      ? undefined
      : standard.accepted
        ? holdout.accepted
          ? 'promotion disabled'
          : 'holdout gate failed'
        : 'standard gate failed';

    result = {
      promoted,
      rejectionReason,
      bestPath: options.bestPath,
      candidatePath,
      candidateMetricsPath,
      summaryPath: options.summaryPath,
      historyPath: options.historyPath,
      standard,
      holdout,
      training: resultTraining
    };
    writeSummary(options.summaryPath, result);
    if (options.historyPath) {
      appendHistory(options.historyPath, result);
    }

    if (promoted) {
      dependencies.beforePromote?.(result);
      ensureParentDirectory(options.bestPath);
      const payload = existsSync(candidatePath)
        ? readFileSync(candidatePath, 'utf8')
        : serializeWeightsPayload(candidateWeights, {
          cycle: 0,
          bestCycle: 0,
          selectionScore: holdout.candidate.score,
          seed: options.seed + iteration - 1,
          replaySamples: 0,
          selfPlaySamples: trained?.samples ?? 0,
          loss: trained?.loss ?? 0
        });
      writeFileSync(options.bestPath, payload, 'utf8');
      currentWeights = candidateWeights;
    }
  }

  if (!result) {
    throw new Error('Promotion loop requires at least one iteration');
  }

  return result;
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const result = runPromotionLoop(parsePromotionLoopArgs(argv));
    console.log(formatGateSummary(result.standard));
    console.log(formatGateSummary(result.holdout));
    console.log(`promoted=${result.promoted}`);
    console.log(`summaryOut=${result.summaryPath}`);
    if (result.promoted) {
      console.log(`weightsOut=${result.bestPath}`);
    } else if (result.rejectionReason) {
      console.log(`rejectionReason=${result.rejectionReason}`);
    }
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

function parseTrainingOptions(argv: readonly string[], seed: number): PolicyGradientCliOptions {
  const trainingArgs = [
    '--native',
    '--seed',
    String(seed),
    '--matches',
    String(positiveIntegerArg(argv, '--matches', 960)),
    '--frames',
    String(positiveIntegerArg(argv, '--frames', 240)),
    '--epochs',
    String(nonNegativeIntegerArg(argv, '--epochs', 2)),
    '--batch-size',
    String(positiveIntegerArg(argv, '--batch-size', 192)),
    '--learning-rate',
    String(numberArg(argv, '--learning-rate', 0.001)),
    '--ppo-clip',
    String(numberArg(argv, '--ppo-clip', 0.12)),
    '--temperature',
    String(numberArg(argv, '--temperature', 1.1)),
    '--discount',
    String(numberArg(argv, '--discount', 0.996)),
    '--start-state-mode',
    stringArg(argv, '--start-state-mode') ?? 'mixed',
    '--advantage-baseline',
    stringArg(argv, '--advantage-baseline') ?? 'learned',
    '--action-mode',
    stringArg(argv, '--action-mode') ?? 'runtime',
    ...(argv.includes('--runtime-survivors-only') ? ['--runtime-survivors-only'] : []),
    '--runtime-wrapper-weight-mode',
    stringArg(argv, '--runtime-wrapper-weight-mode') ?? 'none',
    '--runtime-tactical-rewrite-weight',
    stringArg(argv, '--runtime-tactical-rewrite-weight') ?? '0.5',
    '--action-retention-weight',
    stringArg(argv, '--action-retention-weight') ?? '0',
    '--early-forward-safety-weight',
    stringArg(argv, '--early-forward-safety-weight') ?? '1',
    '--early-forward-anchor-weight',
    stringArg(argv, '--early-forward-anchor-weight') ?? '0',
    '--opponent-mode',
    stringArg(argv, '--opponent-mode') ?? 'league',
    '--league-current-weight',
    stringArg(argv, '--league-current-weight') ?? '1',
    '--league-traditional-weight',
    stringArg(argv, '--league-traditional-weight') ?? '0.15'
  ];
  const nativeBin = stringArg(argv, '--native-bin');
  const openStartRatio = stringArg(argv, '--open-start-ratio');

  if (openStartRatio !== undefined) {
    trainingArgs.push('--open-start-ratio', openStartRatio);
  }
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
  name: 'standard' | 'holdout',
  currentWeights: NeuralWeights,
  candidateWeights: NeuralWeights,
  seeds: readonly number[],
  options: PromotionLoopOptions,
  evaluate: (weights: NeuralWeights, options: RuntimeEvaluationOptions) => PromotionLoopEvaluation
): PromotionLoopGate {
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
  const current = aggregateEvaluations(seedResults.map((row) => row.current));
  const candidate = aggregateEvaluations(seedResults.map((row) => row.candidate));
  const delta = deltaEvaluation(candidate, current);
  const reason = rejectionReason(current, candidate, options);

  return {
    name,
    accepted: reason === undefined,
    reason,
    current,
    candidate,
    delta,
    seeds: seedResults
  };
}

function skippedGate(name: 'holdout', reason: string): PromotionLoopGate {
  const empty = {
    score: 0,
    goalDiff: 0,
    ballProgress: 0,
    goalsFor: 0,
    goalsAgainst: 0,
    winProxy: 0
  };

  return {
    name,
    accepted: false,
    reason,
    current: empty,
    candidate: empty,
    delta: empty,
    seeds: []
  };
}

function aggregateEvaluations(evaluations: readonly PromotionLoopEvaluation[]): PromotionLoopEvaluation {
  if (evaluations.length === 0) {
    return {
      score: 0,
      goalDiff: 0,
      ballProgress: 0,
      goalsFor: 0,
      goalsAgainst: 0,
      winProxy: 0
    };
  }

  const total = evaluations.reduce((sum, row) => ({
    score: sum.score + row.score,
    goalDiff: sum.goalDiff + row.goalDiff,
    ballProgress: sum.ballProgress + row.ballProgress,
    goalsFor: sum.goalsFor + row.goalsFor,
    goalsAgainst: sum.goalsAgainst + row.goalsAgainst,
    winProxy: sum.winProxy + row.winProxy
  }), {
    score: 0,
    goalDiff: 0,
    ballProgress: 0,
    goalsFor: 0,
    goalsAgainst: 0,
    winProxy: 0
  });

  return {
    score: total.score / evaluations.length,
    goalDiff: total.goalsFor - total.goalsAgainst,
    ballProgress: total.ballProgress / evaluations.length,
    goalsFor: total.goalsFor,
    goalsAgainst: total.goalsAgainst,
    winProxy: total.winProxy / evaluations.length
  };
}

function deltaEvaluation(
  candidate: PromotionLoopEvaluation,
  current: PromotionLoopEvaluation
): PromotionLoopEvaluation {
  return {
    score: candidate.score - current.score,
    goalDiff: candidate.goalDiff - current.goalDiff,
    ballProgress: candidate.ballProgress - current.ballProgress,
    goalsFor: candidate.goalsFor - current.goalsFor,
    goalsAgainst: candidate.goalsAgainst - current.goalsAgainst,
    winProxy: candidate.winProxy - current.winProxy
  };
}

function rejectionReason(
  current: PromotionLoopEvaluation,
  candidate: PromotionLoopEvaluation,
  options: PromotionLoopOptions
): string | undefined {
  if (candidate.score <= current.score + options.minScoreDelta) {
    return 'score did not improve';
  }
  if (candidate.goalDiff < current.goalDiff - options.maxGoalDiffRegression) {
    return 'goal differential regressed';
  }
  if (candidate.winProxy < current.winProxy - options.maxWinProxyRegression) {
    return 'win proxy regressed';
  }
  return undefined;
}

function writeSummary(path: string, result: PromotionLoopResult): void {
  ensureParentDirectory(path);
  writeFileSync(path, `${JSON.stringify({
    generatedAt: new Date().toISOString(),
    promoted: result.promoted,
    rejectionReason: result.rejectionReason,
    bestPath: result.bestPath,
    candidatePath: result.candidatePath,
    candidateMetricsPath: result.candidateMetricsPath,
    historyPath: result.historyPath,
    training: result.training,
    gates: {
      standard: result.standard,
      holdout: result.holdout
    }
  }, null, 2)}\n`, 'utf8');
}

function appendHistory(path: string, result: PromotionLoopResult): void {
  ensureParentDirectory(path);
  appendFileSync(path, `${JSON.stringify({
    generatedAt: new Date().toISOString(),
    promoted: result.promoted,
    rejectionReason: result.rejectionReason,
    seed: result.training.seed,
    candidatePath: result.candidatePath,
    candidateMetricsPath: result.candidateMetricsPath,
    standardAccepted: result.standard.accepted,
    holdoutAccepted: result.holdout.accepted,
    standardScoreDelta: result.standard.delta.score,
    holdoutScoreDelta: result.holdout.delta.score,
    standardGoals: `${result.standard.candidate.goalsFor}-${result.standard.candidate.goalsAgainst}`,
    holdoutGoals: `${result.holdout.candidate.goalsFor}-${result.holdout.candidate.goalsAgainst}`,
    advantageBaseline: result.training.advantageBaseline,
    opponentMode: result.training.opponentMode,
    startStateMode: result.training.startStateMode,
    openStartRatio: result.training.openStartRatio,
    actionMode: result.training.actionMode,
    runtimeSurvivorsOnly: result.training.runtimeSurvivorsOnly,
    runtimeWrapperWeightMode: result.training.runtimeWrapperWeightMode,
    runtimeTacticalRewriteWeight: result.training.runtimeTacticalRewriteWeight,
    actionRetentionWeight: result.training.actionRetentionWeight,
    earlyForwardSafetyWeight: result.training.earlyForwardSafetyWeight,
    earlyForwardAnchorWeight: result.training.earlyForwardAnchorWeight,
    matches: result.training.matches,
    frames: result.training.frames,
    epochs: result.training.epochs,
    batchSize: result.training.batchSize,
    learningRate: result.training.learningRate
  })}\n`, 'utf8');
}

function requiredCandidateInputPath(options: PromotionLoopOptions): string {
  if (!options.candidateInputPath) {
    throw new Error('Missing candidate input path');
  }
  return options.candidateInputPath;
}

function trainingFromCandidateMetadata(
  fallback: PolicyGradientCliOptions,
  json: string
): PolicyGradientCliOptions {
  const metadata = candidateMetadata(json);
  if (!metadata) {
    return fallback;
  }

  return {
    ...fallback,
    seed: finiteMetadataNumber(metadata, 'seed', fallback.seed),
    matches: finiteMetadataNumber(metadata, 'matches', fallback.matches),
    frames: finiteMetadataNumber(metadata, 'frames', fallback.frames),
    epochs: finiteMetadataNumber(metadata, 'epochs', fallback.epochs),
    batchSize: finiteMetadataNumber(metadata, 'batchSize', fallback.batchSize),
    learningRate: finiteMetadataNumber(metadata, 'learningRate', fallback.learningRate),
    ppoClip: finiteMetadataNumber(metadata, 'ppoClip', fallback.ppoClip),
    temperature: finiteMetadataNumber(metadata, 'temperature', fallback.temperature),
    openStartRatio: optionalFiniteMetadataNumber(metadata, 'openStartRatio', fallback.openStartRatio),
    discount: finiteMetadataNumber(metadata, 'discount', fallback.discount),
    startStateMode: startStateModeMetadata(metadata, fallback.startStateMode),
    advantageBaseline: advantageBaselineMetadata(metadata, fallback.advantageBaseline),
    actionMode: actionModeMetadata(metadata, fallback.actionMode),
    runtimeSurvivorsOnly: booleanMetadata(metadata, 'runtimeSurvivorsOnly', fallback.runtimeSurvivorsOnly),
    runtimeWrapperWeightMode: runtimeWrapperWeightModeMetadata(metadata, fallback.runtimeWrapperWeightMode),
    runtimeTacticalRewriteWeight: finiteMetadataNumber(
      metadata,
      'runtimeTacticalRewriteWeight',
      fallback.runtimeTacticalRewriteWeight
    ),
    actionRetentionWeight: finiteMetadataNumber(metadata, 'actionRetentionWeight', fallback.actionRetentionWeight),
    earlyForwardSafetyWeight: finiteMetadataNumber(
      metadata,
      'earlyForwardSafetyWeight',
      fallback.earlyForwardSafetyWeight
    ),
    earlyForwardAnchorWeight: finiteMetadataNumber(
      metadata,
      'earlyForwardAnchorWeight',
      fallback.earlyForwardAnchorWeight
    ),
    opponentMode: opponentModeMetadata(metadata, fallback.opponentMode)
  };
}

function candidateMetadata(json: string): Record<string, unknown> | undefined {
  const parsed = JSON.parse(json) as unknown;
  return isRecord(parsed) && isRecord(parsed.metadata) ? parsed.metadata : undefined;
}

function finiteMetadataNumber(
  metadata: Record<string, unknown>,
  field: string,
  fallback: number
): number {
  const value = metadata[field];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionalFiniteMetadataNumber(
  metadata: Record<string, unknown>,
  field: string,
  fallback: number | undefined
): number | undefined {
  const value = metadata[field];
  return typeof value === 'number' && Number.isFinite(value) ? clamp01(value) : fallback;
}

function booleanMetadata(
  metadata: Record<string, unknown>,
  field: string,
  fallback: boolean
): boolean {
  const value = metadata[field];
  return typeof value === 'boolean' ? value : fallback;
}

function startStateModeMetadata(
  metadata: Record<string, unknown>,
  fallback: PolicyGradientCliOptions['startStateMode']
): PolicyGradientCliOptions['startStateMode'] {
  const value = metadata.startStateMode;
  return value === 'open' ||
    value === 'outcome-curriculum' ||
    value === 'own-goal-defense' ||
    value === 'corner-fight' ||
    value === 'loose-ball-contest' ||
    value === 'mixed'
    ? value
    : fallback;
}

function advantageBaselineMetadata(
  metadata: Record<string, unknown>,
  fallback: PolicyGradientCliOptions['advantageBaseline']
): PolicyGradientCliOptions['advantageBaseline'] {
  const value = metadata.advantageBaseline;
  return value === 'global' || value === 'start-team-time' || value === 'learned'
    ? value
    : fallback;
}

function actionModeMetadata(
  metadata: Record<string, unknown>,
  fallback: PolicyGradientCliOptions['actionMode']
): PolicyGradientCliOptions['actionMode'] {
  const value = metadata.actionMode;
  return value === 'raw' || value === 'runtime' ? value : fallback;
}

function runtimeWrapperWeightModeMetadata(
  metadata: Record<string, unknown>,
  fallback: PolicyGradientCliOptions['runtimeWrapperWeightMode']
): PolicyGradientCliOptions['runtimeWrapperWeightMode'] {
  const value = metadata.runtimeWrapperWeightMode;
  return value === 'none' || value === 'tactical-downweight' ? value : fallback;
}

function opponentModeMetadata(
  metadata: Record<string, unknown>,
  fallback: PolicyGradientCliOptions['opponentMode']
): PolicyGradientCliOptions['opponentMode'] {
  const value = metadata.opponentMode;
  return value === 'self' || value === 'traditional' || value === 'league' ? value : fallback;
}

function formatGateSummary(gate: PromotionLoopGate): string {
  return [
    `gate=${gate.name}`,
    `accepted=${gate.accepted}`,
    `currentGoals=${gate.current.goalsFor}-${gate.current.goalsAgainst}`,
    `candidateGoals=${gate.candidate.goalsFor}-${gate.candidate.goalsAgainst}`,
    `currentAvgScore=${gate.current.score.toFixed(3)}`,
    `candidateAvgScore=${gate.candidate.score.toFixed(3)}`,
    `currentAvgWin=${gate.current.winProxy.toFixed(3)}`,
    `candidateAvgWin=${gate.candidate.winProxy.toFixed(3)}`,
    `currentAvgBp=${gate.current.ballProgress.toFixed(3)}`,
    `candidateAvgBp=${gate.candidate.ballProgress.toFixed(3)}`,
    gate.reason ? `reason=${gate.reason}` : undefined
  ].filter((part): part is string => part !== undefined).join(' ');
}

function ensureParentDirectory(path: string): void {
  const parent = dirname(path);
  if (parent && parent !== '.' && !existsSync(parent)) {
    mkdirSync(parent, { recursive: true });
  }
}

function iterationPath(path: string, iteration: number, iterations: number): string {
  if (iterations === 1) {
    return path;
  }

  const extensionIndex = path.lastIndexOf('.');
  if (extensionIndex === -1) {
    return `${path}-i${iteration}`;
  }
  return `${path.slice(0, extensionIndex)}-i${iteration}${path.slice(extensionIndex)}`;
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

function clamp01(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
}

function integerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.floor(numberArg(argv, name, fallback));
}

function positiveIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.max(1, integerArg(argv, name, fallback));
}

function nonNegativeIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.max(0, integerArg(argv, name, fallback));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/promote-policy-gradient.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/promote-policy-gradient.js')) {
  main();
}
