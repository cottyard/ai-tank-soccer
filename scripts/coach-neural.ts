import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNeuralStrategy } from '../src/ai/neuralStrategy';
import { evaluateNeuralWeights } from '../src/ai/neuralTraining';
import { NEURAL_WEIGHT_COUNT, defaultNeuralWeights, type NeuralWeights } from '../src/ai/neuralWeights';
import { LearningReplayBuffer, trainOfflineFromReplay, type LearningSample, type LearningTag } from '../src/ai/imitationLearning';
import { POLICY_ACTION_COUNT } from '../src/ai/policyActions';
import { POLICY_INPUT_COUNT } from '../src/ai/policyNetwork';
import { trainSelfPlayPolicy } from '../src/ai/selfPlayTraining';
import {
  trainPolicyGradientSelfPlay,
  type PolicyGradientAdvantageBaseline,
  type PolicyGradientStartStateMode
} from '../src/ai/policyGradientTraining';
import { trainCurriculumPolicy } from '../src/ai/curriculumTraining';
import { evaluateRuntimePolicy } from '../src/ai/policyGate';
import { traditionalStrategy } from '../src/ai/traditionalStrategy';
import { FIELD, createInitialState, type GameState, type Team } from '../src/game/model';
import { simulateMatch } from '../src/game/match';
import { idleCommands, type Strategy } from '../src/game/strategy';

declare const process: {
  argv: string[];
  cwd(): string;
  exitCode?: number;
  platform: string;
};

type OpponentName = 'idle' | 'traditional' | 'neural-default' | 'neural-self' | 'neural-current';
type MetricPhase = 'eval' | 'train';
type AcceptOpponentName = OpponentName | 'league' | 'runtime';

export type CoachOptions = {
  seed: number;
  cycles: number;
  evalMatches: number;
  frames: number;
  selfPlayMatches: number;
  selfPlayFrames: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  exploration: number;
  rlCycles: number;
  rlMatches: number;
  rlFrames: number;
  rlEpochs: number;
  rlBatchSize: number;
  rlLearningRate: number;
  rlPpoClip: number;
  rlTemperature: number;
  rlDiscount: number;
  rlStartStateMode: PolicyGradientStartStateMode;
  rlAdvantageBaseline: PolicyGradientAdvantageBaseline;
  rlActionMode: 'raw' | 'runtime';
  rlOpponentMode: 'self' | 'traditional' | 'league';
  rlNative: boolean;
  rlNativeBin?: string;
  curriculumScenarios: number;
  curriculumFrames: number;
  input?: string;
  replay?: string;
  output?: string;
  printWeights: boolean;
  acceptOpponent?: AcceptOpponentName;
  gateSeeds: number;
  runtimeGateMatches: number;
  runtimeGateFrames: number;
};

export type EvaluationSuiteOptions = {
  seed: number;
  matches: number;
  frames: number;
  fixedNeuralOpponentWeights?: NeuralWeights;
};

export type EvaluationRow = {
  opponent: OpponentName;
  score: number;
  goalDiff: number;
  goalsFor: number;
  goalsAgainst: number;
  ballProgress: number;
  winProxy: number;
};

export type CycleMetric = EvaluationRow & {
  cycle: number;
  phase: MetricPhase;
  replaySamples: number;
  selfPlaySamples: number;
  loss: number;
  accepted?: boolean;
  acceptOpponent?: AcceptOpponentName;
  acceptScore?: number;
  acceptBaseline?: number;
};

export type WeightsMetadata = {
  cycle: number;
  bestCycle: number;
  selectionScore: number;
  seed: number;
  replaySamples: number;
  selfPlaySamples: number;
  loss: number;
  acceptOpponent?: AcceptOpponentName;
  gateSeeds?: number;
  runtimeGateMatches?: number;
  runtimeGateFrames?: number;
  acceptedCycles?: number;
  rejectedCycles?: number;
  replayAccepted?: boolean;
  replayScore?: number;
  replayBaseline?: number;
};

type Opponent = {
  name: OpponentName;
  strategy: Strategy;
};

type MatchTotals = {
  goalsFor: number;
  goalsAgainst: number;
  ballProgress: number;
  wins: number;
};

const idleStrategy: Strategy = {
  name: 'idle',
  decide: idleCommands
};

const DEFAULT_OPTIONS: CoachOptions = {
  seed: 1,
  cycles: 2,
  evalMatches: 2,
  frames: 30 * 12,
  selfPlayMatches: 4,
  selfPlayFrames: 30 * 12,
  epochs: 12,
  batchSize: 32,
  learningRate: 0.018,
  exploration: 0.18,
  rlCycles: 0,
  rlMatches: 8,
  rlFrames: 30 * 20,
  rlEpochs: 3,
  rlBatchSize: 64,
  rlLearningRate: 0.006,
  rlPpoClip: 0.2,
  rlTemperature: 1.08,
  rlDiscount: 0.992,
  rlStartStateMode: 'outcome-curriculum',
  rlAdvantageBaseline: 'global',
  rlActionMode: 'raw',
  rlOpponentMode: 'self',
  rlNative: false,
  curriculumScenarios: 0,
  curriculumFrames: 14,
  printWeights: false,
  gateSeeds: 1,
  runtimeGateMatches: 0,
  runtimeGateFrames: 0
};

export function parseCoachArgs(argv: readonly string[]): CoachOptions {
  return {
    seed: numberArg(argv, '--seed', DEFAULT_OPTIONS.seed),
    cycles: nonNegativeIntegerArg(argv, '--cycles', DEFAULT_OPTIONS.cycles),
    evalMatches: positiveIntegerArg(argv, '--eval-matches', DEFAULT_OPTIONS.evalMatches),
    frames: positiveIntegerArg(argv, '--frames', DEFAULT_OPTIONS.frames),
    selfPlayMatches: positiveIntegerArg(argv, '--self-play-matches', DEFAULT_OPTIONS.selfPlayMatches),
    selfPlayFrames: positiveIntegerArg(argv, '--self-play-frames', DEFAULT_OPTIONS.selfPlayFrames),
    epochs: nonNegativeIntegerArg(argv, '--epochs', DEFAULT_OPTIONS.epochs),
    batchSize: positiveIntegerArg(argv, '--batch-size', DEFAULT_OPTIONS.batchSize),
    learningRate: numberArg(argv, '--learning-rate', DEFAULT_OPTIONS.learningRate),
    exploration: clamp01(numberArg(argv, '--exploration', DEFAULT_OPTIONS.exploration)),
    rlCycles: nonNegativeIntegerArg(argv, '--rl-cycles', DEFAULT_OPTIONS.rlCycles),
    rlMatches: positiveIntegerArg(argv, '--rl-matches', DEFAULT_OPTIONS.rlMatches),
    rlFrames: positiveIntegerArg(argv, '--rl-frames', DEFAULT_OPTIONS.rlFrames),
    rlEpochs: nonNegativeIntegerArg(argv, '--rl-epochs', DEFAULT_OPTIONS.rlEpochs),
    rlBatchSize: positiveIntegerArg(argv, '--rl-batch-size', DEFAULT_OPTIONS.rlBatchSize),
    rlLearningRate: numberArg(argv, '--rl-learning-rate', DEFAULT_OPTIONS.rlLearningRate),
    rlPpoClip: Math.max(0, numberArg(argv, '--rl-ppo-clip', DEFAULT_OPTIONS.rlPpoClip)),
    rlTemperature: numberArg(argv, '--rl-temperature', DEFAULT_OPTIONS.rlTemperature),
    rlDiscount: clamp01(numberArg(argv, '--rl-discount', DEFAULT_OPTIONS.rlDiscount)),
    rlStartStateMode: startStateModeArg(argv, '--rl-start-state-mode', DEFAULT_OPTIONS.rlStartStateMode),
    rlAdvantageBaseline: advantageBaselineArg(argv, '--rl-advantage-baseline', DEFAULT_OPTIONS.rlAdvantageBaseline),
    rlActionMode: actionModeArg(argv, '--rl-action-mode', DEFAULT_OPTIONS.rlActionMode),
    rlOpponentMode: opponentModeArg(argv, '--rl-opponent-mode', DEFAULT_OPTIONS.rlOpponentMode),
    rlNative: argv.includes('--rl-native'),
    rlNativeBin: stringArg(argv, '--rl-native-bin'),
    curriculumScenarios: nonNegativeIntegerArg(argv, '--curriculum-scenarios', DEFAULT_OPTIONS.curriculumScenarios),
    curriculumFrames: positiveIntegerArg(argv, '--curriculum-frames', DEFAULT_OPTIONS.curriculumFrames),
    input: stringArg(argv, '--input'),
    replay: stringArg(argv, '--replay'),
    output: stringArg(argv, '--output'),
    printWeights: argv.includes('--print-weights'),
    acceptOpponent: opponentArg(argv, '--accept-opponent'),
    gateSeeds: positiveIntegerArg(argv, '--gate-seeds', DEFAULT_OPTIONS.gateSeeds),
    runtimeGateMatches: nonNegativeIntegerArg(argv, '--runtime-gate-matches', DEFAULT_OPTIONS.runtimeGateMatches),
    runtimeGateFrames: nonNegativeIntegerArg(argv, '--runtime-gate-frames', DEFAULT_OPTIONS.runtimeGateFrames)
  };
}

export function runEvaluationSuite(
  weights: NeuralWeights,
  options: EvaluationSuiteOptions
): EvaluationRow[] {
  validateWeights(weights);

  return createOpponents(weights, options.fixedNeuralOpponentWeights).map((opponent, index) => {
    const tactical = evaluateNeuralWeights(weights, {
      seed: options.seed + index * 104_729,
      opponent: opponent.strategy,
      matches: options.matches,
      frames: options.frames
    });
    const matchTotals = runSeededMatches(weights, opponent, {
      seed: options.seed + index * 8_191,
      matches: options.matches,
      frames: options.frames
    });

    return {
      opponent: opponent.name,
      score: tactical.score,
      goalDiff: tactical.goalDiff,
      goalsFor: matchTotals.goalsFor,
      goalsAgainst: matchTotals.goalsAgainst,
      ballProgress: tactical.ballProgress,
      winProxy: matchTotals.wins / options.matches
    };
  });
}

export function formatCycleMetric(metric: CycleMetric): string {
  const parts = [
    `cycle=${metric.cycle}`,
    `phase=${metric.phase}`,
    `opponent=${metric.opponent}`,
    `score=${metric.score.toFixed(3)}`,
    `goalDiff=${metric.goalDiff.toFixed(3)}`,
    `goals=${metric.goalsFor}-${metric.goalsAgainst}`,
    `ballProgress=${metric.ballProgress.toFixed(3)}`,
    `winProxy=${metric.winProxy.toFixed(3)}`,
    `replaySamples=${metric.replaySamples}`,
    `selfPlaySamples=${metric.selfPlaySamples}`,
    `loss=${metric.loss.toFixed(3)}`
  ];

  if (metric.accepted !== undefined) {
    parts.push(`accepted=${metric.accepted}`);
  }
  if (metric.acceptOpponent) {
    parts.push(`acceptOpponent=${metric.acceptOpponent}`);
  }
  if (metric.acceptScore !== undefined) {
    parts.push(`acceptScore=${metric.acceptScore.toFixed(3)}`);
  }
  if (metric.acceptBaseline !== undefined) {
    parts.push(`acceptBaseline=${metric.acceptBaseline.toFixed(3)}`);
  }
  if (metric.acceptScore !== undefined && metric.acceptBaseline !== undefined) {
    parts.push(`acceptDelta=${(metric.acceptScore - metric.acceptBaseline).toFixed(3)}`);
  }

  return parts.join(' ');
}

export function loadWeightsPayload(json: string): number[] {
  const parsed = JSON.parse(json) as unknown;
  const weights = Array.isArray(parsed)
    ? parsed
    : isRecord(parsed) && Array.isArray(parsed.weights)
      ? parsed.weights
      : null;

  if (!weights) {
    throw new Error('Expected weights JSON as an array or an object with a weights array');
  }

  return validateWeights(weights);
}

export function loadReplayPayload(json: string): LearningSample[] {
  const parsed = JSON.parse(json) as unknown;
  const samples = Array.isArray(parsed)
    ? parsed
    : isRecord(parsed) && Array.isArray(parsed.samples)
      ? parsed.samples
      : null;

  if (!samples) {
    throw new Error('Expected replay JSON as an array or an object with a samples array');
  }

  return samples.map((sample, index) => validateReplaySample(sample, index));
}

export function serializeWeightsPayload(weights: NeuralWeights, metadata: WeightsMetadata): string {
  return `${JSON.stringify(
    {
      weights: validateWeights(weights),
      metadata
    },
    null,
    2
  )}\n`;
}

export function runCoach(options: CoachOptions): {
  weights: number[];
  metrics: CycleMetric[];
  metadata: WeightsMetadata;
} {
  let weights = options.input
    ? loadWeightsPayload(readFileSync(options.input, 'utf8'))
    : defaultNeuralWeights();
  const metrics: CycleMetric[] = [];
  let replaySamples = 0;
  let selfPlaySamples = 0;
  let loss = 0;
  let bestWeights = [...weights];
  let bestCycle = 0;
  let bestSelectionScore = Number.NEGATIVE_INFINITY;
  let acceptedCycles = 0;
  let rejectedCycles = 0;
  let replayAccepted: boolean | undefined;
  let replayScore: number | undefined;
  let replayBaseline: number | undefined;

  if (options.replay) {
    const replay = new LearningReplayBuffer();
    const loadedReplay = loadReplayPayload(readFileSync(options.replay, 'utf8'));
    const beforeReplayWeights = [...weights];
    replayBaseline = acceptanceScore(beforeReplayWeights, options, 0, beforeReplayWeights);
    replay.load(loadedReplay);
    const replayTraining = trainOfflineFromReplay(weights, replay, {
      epochs: 1,
      batchSize: options.batchSize,
      learningRate: Math.min(options.learningRate, 0.0015),
      l2: 0.00024,
      gradientClip: 1.4,
      seed: options.seed + 4049
    });
    replayScore = acceptanceScore(replayTraining.weights, options, 0, beforeReplayWeights);
    replayAccepted = replayScore === undefined || replayBaseline === undefined || replayScore > replayBaseline;
    weights = replayAccepted ? replayTraining.weights : beforeReplayWeights;
    if (replayAccepted) {
      bestWeights = [...weights];
    }
    loss = replayTraining.loss;
    replaySamples += replayTraining.trainedSamples;
  }

  const initialRows = pushEvaluationMetrics(metrics, 0, weights, options, replaySamples, selfPlaySamples, loss);
  bestSelectionScore = selectionScore(initialRows);
  bestWeights = [...weights];

  if (options.curriculumScenarios > 0) {
    const gateOpponentWeights = [...weights];
    const trained = trainCurriculumPolicy({
      weights,
      scenarios: options.curriculumScenarios,
      rolloutFrames: options.curriculumFrames,
      epochs: options.epochs,
      batchSize: options.batchSize,
      learningRate: Math.min(options.learningRate, 0.004),
      seed: options.seed + 11_317
    });
    const baselineScore = acceptanceScore(weights, options, 0, gateOpponentWeights);
    const candidateScore = acceptanceScore(trained.weights, options, 0, gateOpponentWeights);
    const accepted = candidateScore === undefined || baselineScore === undefined || candidateScore > baselineScore;
    if (accepted) {
      weights = trained.weights;
      bestWeights = [...weights];
      acceptedCycles += options.acceptOpponent ? 1 : 0;
    } else {
      rejectedCycles += 1;
    }
    replaySamples += trained.trainedSamples;
    selfPlaySamples += trained.samples;
    loss = trained.loss;
    metrics.push({
      cycle: 0,
      phase: 'train',
      opponent: 'neural-self',
      score: 0,
      goalDiff: 0,
      goalsFor: 0,
      goalsAgainst: 0,
      ballProgress: 0,
      winProxy: 0.5,
      replaySamples,
      selfPlaySamples,
      loss,
      accepted: options.acceptOpponent ? accepted : undefined,
      acceptOpponent: options.acceptOpponent,
      acceptScore: candidateScore,
      acceptBaseline: baselineScore
    });
  }

  for (let cycle = 1; cycle <= options.cycles; cycle += 1) {
    const gateOpponentWeights = [...weights];
    const trained = trainSelfPlayPolicy({
      weights,
      matches: options.selfPlayMatches,
      frames: options.selfPlayFrames,
      epochs: options.epochs,
      batchSize: options.batchSize,
      learningRate: options.learningRate,
      exploration: options.exploration,
      seed: options.seed + cycle * 65_537
    });
    replaySamples += trained.trainedSamples;
    selfPlaySamples += trained.samples;
    loss = trained.loss;

    const baselineScore = acceptanceScore(weights, options, cycle, gateOpponentWeights);
    const candidateScore = acceptanceScore(trained.weights, options, cycle, gateOpponentWeights);
    const accepted = candidateScore === undefined || baselineScore === undefined || candidateScore > baselineScore;
    if (accepted) {
      weights = trained.weights;
      bestWeights = [...weights];
      acceptedCycles += options.acceptOpponent ? 1 : 0;
    } else {
      rejectedCycles += 1;
    }

    metrics.push({
      cycle,
      phase: 'train',
      opponent: 'neural-self',
      score: trained.redGoals - trained.blueGoals,
      goalDiff: trained.redGoals - trained.blueGoals,
      goalsFor: trained.redGoals,
      goalsAgainst: trained.blueGoals,
      ballProgress: (trained.finalState.ball.position.x - FIELD.length / 2) / FIELD.length,
      winProxy: trained.redGoals > trained.blueGoals ? 1 : trained.redGoals === trained.blueGoals ? 0.5 : 0,
      replaySamples,
      selfPlaySamples,
      loss,
      accepted: options.acceptOpponent ? accepted : undefined,
      acceptOpponent: options.acceptOpponent,
      acceptScore: candidateScore,
      acceptBaseline: baselineScore
    });

    const rows = pushEvaluationMetrics(metrics, cycle, weights, options, replaySamples, selfPlaySamples, loss);
    const score = selectionScore(rows);
    if (accepted && score > bestSelectionScore) {
      bestWeights = [...weights];
      bestCycle = cycle;
      bestSelectionScore = score;
    }
  }

  for (let rlCycle = 1; rlCycle <= options.rlCycles; rlCycle += 1) {
    const cycle = options.cycles + rlCycle;
    const gateOpponentWeights = [...weights];
    const trained = trainPolicyGradientCycle(weights, chooseSelfPlayOpponent(weights, bestWeights, rlCycle), options, rlCycle);
    selfPlaySamples += trained.samples;
    loss = trained.loss;

    const baselineScore = acceptanceScore(weights, options, cycle, gateOpponentWeights);
    const candidateScore = acceptanceScore(trained.weights, options, cycle, gateOpponentWeights);
    const accepted = candidateScore === undefined || baselineScore === undefined || candidateScore > baselineScore;
    if (accepted) {
      weights = trained.weights;
      bestWeights = [...weights];
      acceptedCycles += options.acceptOpponent ? 1 : 0;
    } else {
      rejectedCycles += 1;
    }

    metrics.push({
      cycle,
      phase: 'train',
      opponent: 'neural-self',
      score: trained.redGoals - trained.blueGoals,
      goalDiff: trained.redGoals - trained.blueGoals,
      goalsFor: trained.redGoals,
      goalsAgainst: trained.blueGoals,
      ballProgress: (trained.finalState.ball.position.x - FIELD.length / 2) / FIELD.length,
      winProxy: trained.redGoals > trained.blueGoals ? 1 : trained.redGoals === trained.blueGoals ? 0.5 : 0,
      replaySamples,
      selfPlaySamples,
      loss,
      accepted: options.acceptOpponent ? accepted : undefined,
      acceptOpponent: options.acceptOpponent,
      acceptScore: candidateScore,
      acceptBaseline: baselineScore
    });

    const rows = pushEvaluationMetrics(metrics, cycle, weights, options, replaySamples, selfPlaySamples, loss);
    const score = selectionScore(rows);
    if (accepted && score > bestSelectionScore) {
      bestWeights = [...weights];
      bestCycle = cycle;
      bestSelectionScore = score;
    }
  }

  const metadata = {
    cycle: bestCycle,
    bestCycle,
    selectionScore: bestSelectionScore,
    seed: options.seed,
    replaySamples,
    selfPlaySamples,
    loss,
    acceptOpponent: options.acceptOpponent,
    gateSeeds: options.gateSeeds,
    runtimeGateMatches: runtimeGateMatches(options),
    runtimeGateFrames: runtimeGateFrames(options),
    acceptedCycles,
    rejectedCycles,
    replayAccepted,
    replayScore,
    replayBaseline
  };

  if (options.output) {
    writeFileSync(options.output, serializeWeightsPayload(bestWeights, metadata), 'utf8');
  }

  return { weights: bestWeights, metrics, metadata };
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const options = parseCoachArgs(argv);
    const result = runCoach(options);
    for (const metric of result.metrics) {
      console.log(formatCycleMetric(metric));
    }
    if (options.output) {
      console.log(`weightsOut=${options.output}`);
    }
    if (options.printWeights) {
      console.log(`weights=${JSON.stringify(result.weights)}`);
    }
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

function pushEvaluationMetrics(
  metrics: CycleMetric[],
  cycle: number,
  weights: NeuralWeights,
  options: CoachOptions,
  replaySamples: number,
  selfPlaySamples: number,
  loss: number
): EvaluationRow[] {
  const rows = runEvaluationSuite(weights, {
    seed: options.seed + cycle * 16_381,
    matches: options.evalMatches,
    frames: options.frames
  });

  for (const row of rows) {
    metrics.push({
      cycle,
      phase: 'eval',
      ...row,
      replaySamples,
      selfPlaySamples,
      loss
    });
  }

  return rows;
}

function selectionScore(rows: readonly EvaluationRow[]): number {
  return rows.reduce((total, row) => total + rowSelectionScore(row), 0);
}

function rowSelectionScore(row: EvaluationRow): number {
  const weights: Record<OpponentName, number> = {
    idle: 0.2,
    traditional: 2.4,
    'neural-default': 0.9,
    'neural-self': 0.5,
    'neural-current': 1.8
  };

  const matchSignal = row.winProxy * 40 + (row.goalsFor - row.goalsAgainst) * 55 + row.ballProgress * 80;
  return weights[row.opponent] * (row.score + matchSignal);
}

function acceptanceScore(
  weights: NeuralWeights,
  options: CoachOptions,
  cycle: number,
  fixedNeuralOpponentWeights: NeuralWeights = weights
): number | undefined {
  if (!options.acceptOpponent) {
    return undefined;
  }

  let total = 0;
  for (let gateSeed = 0; gateSeed < options.gateSeeds; gateSeed += 1) {
    const seed = options.seed + cycle * 16_381 + gateSeed * 1_000_003;
    if (options.acceptOpponent === 'runtime') {
      total += evaluateRuntimePolicy(weights, {
        seed,
        matches: runtimeGateMatches(options),
        frames: runtimeGateFrames(options)
      }).score;
      continue;
    }

    const rows = runEvaluationSuite(weights, {
      seed,
      matches: options.evalMatches,
      frames: options.frames,
      fixedNeuralOpponentWeights
    });
    total += options.acceptOpponent === 'league'
      ? leagueSelectionScore(rows)
      : opponentSelectionScore(rows, options.acceptOpponent);
  }

  return total / options.gateSeeds;
}

function opponentSelectionScore(rows: readonly EvaluationRow[], opponent: OpponentName): number {
  const row = rows.find((candidate) => candidate.opponent === opponent);
  if (!row) {
    throw new Error(`Missing acceptance opponent ${opponent}`);
  }
  return rowSelectionScore(row);
}

function leagueSelectionScore(rows: readonly EvaluationRow[]): number {
  return (
    opponentSelectionScore(rows, 'traditional') * 0.48 +
    opponentSelectionScore(rows, 'neural-current') * 0.42 +
    opponentSelectionScore(rows, 'neural-default') * 0.1
  );
}

function runtimeGateMatches(options: CoachOptions): number {
  return options.runtimeGateMatches > 0 ? options.runtimeGateMatches : options.evalMatches;
}

function runtimeGateFrames(options: CoachOptions): number {
  return options.runtimeGateFrames > 0 ? options.runtimeGateFrames : options.frames;
}

function runSeededMatches(
  weights: NeuralWeights,
  opponent: Opponent,
  options: EvaluationSuiteOptions
): MatchTotals {
  let goalsFor = 0;
  let goalsAgainst = 0;
  let ballProgress = 0;
  let wins = 0;

  for (let match = 0; match < options.matches; match += 1) {
    const candidateTeam: Team = match % 2 === 0 ? 'red' : 'blue';
    const candidate = createNeuralStrategy({
      weights,
      name: 'neural-candidate',
      tacticalRollout: false
    });
    const initialState = createSeededInitialState(options.seed, match, candidateTeam);
    const result = simulateMatch({
      red: candidateTeam === 'red' ? candidate : opponent.strategy,
      blue: candidateTeam === 'blue' ? candidate : opponent.strategy,
      frames: options.frames,
      initialState
    });
    const forGoals = candidateTeam === 'red' ? result.state.score.red : result.state.score.blue;
    const againstGoals = candidateTeam === 'red' ? result.state.score.blue : result.state.score.red;

    goalsFor += forGoals;
    goalsAgainst += againstGoals;
    ballProgress += attackProgress(result.state, initialState, candidateTeam);
    wins += forGoals > againstGoals ? 1 : forGoals === againstGoals ? 0.5 : 0;
  }

  return {
    goalsFor,
    goalsAgainst,
    ballProgress: ballProgress / options.matches,
    wins
  };
}

function createOpponents(candidateWeights: NeuralWeights, fixedNeuralOpponentWeights?: NeuralWeights): Opponent[] {
  const opponents: Opponent[] = [
    { name: 'idle', strategy: idleStrategy },
    { name: 'traditional', strategy: traditionalStrategy },
    {
      name: 'neural-default',
      strategy: createNeuralStrategy({
        weights: defaultNeuralWeights(),
        name: 'neural-default',
        tacticalRollout: false
      })
    },
    {
      name: 'neural-self',
      strategy: createNeuralStrategy({
        weights: candidateWeights,
        name: 'neural-self',
        tacticalRollout: false
      })
    }
  ];

  if (fixedNeuralOpponentWeights) {
    opponents.push({
      name: 'neural-current',
      strategy: createNeuralStrategy({
        weights: fixedNeuralOpponentWeights,
        name: 'neural-current',
        tacticalRollout: false
      })
    });
  }

  return opponents;
}

function chooseSelfPlayOpponent(
  currentWeights: NeuralWeights,
  bestWeights: NeuralWeights,
  cycle: number
): NeuralWeights {
  return cycle % 3 === 0 ? bestWeights : currentWeights;
}

function trainPolicyGradientCycle(
  weights: NeuralWeights,
  opponentWeights: NeuralWeights,
  options: CoachOptions,
  rlCycle: number
): ReturnType<typeof trainPolicyGradientSelfPlay> {
  const seed = options.seed + rlCycle * 1_000_033;
  if (!options.rlNative) {
    return trainPolicyGradientSelfPlay({
      weights,
      opponentWeights,
      matches: options.rlMatches,
      frames: options.rlFrames,
      epochs: options.rlEpochs,
      batchSize: options.rlBatchSize,
      learningRate: options.rlLearningRate,
      ppoClip: options.rlPpoClip,
      temperature: options.rlTemperature,
      discount: options.rlDiscount,
      advantageBaseline: options.rlAdvantageBaseline,
      startStateMode: options.rlStartStateMode,
      seed
    });
  }

  return runNativePolicyGradientCycle(weights, opponentWeights, {
    seed,
    matches: options.rlMatches,
    frames: options.rlFrames,
    epochs: options.rlEpochs,
    batchSize: options.rlBatchSize,
    learningRate: options.rlLearningRate,
    ppoClip: options.rlPpoClip,
    temperature: options.rlTemperature,
    discount: options.rlDiscount,
    advantageBaseline: options.rlAdvantageBaseline,
    startStateMode: options.rlStartStateMode,
    actionMode: options.rlActionMode,
    opponentMode: options.rlOpponentMode,
    nativeBin: options.rlNativeBin
  });
}

function runNativePolicyGradientCycle(
  weights: NeuralWeights,
  opponentWeights: NeuralWeights,
  options: {
    seed: number;
    matches: number;
    frames: number;
    epochs: number;
    batchSize: number;
    learningRate: number;
    ppoClip: number;
    temperature: number;
    discount: number;
    startStateMode: PolicyGradientStartStateMode;
    advantageBaseline: PolicyGradientAdvantageBaseline;
    actionMode: 'raw' | 'runtime';
    opponentMode: 'self' | 'traditional' | 'league';
    nativeBin?: string;
  }
): ReturnType<typeof trainPolicyGradientSelfPlay> {
  const nativeBin = resolveNativeTrainer(options.nativeBin);
  const workdir = mkdtempSync(join(tmpdir(), 'soccer-coach-native-rl-'));
  const weightsPath = join(workdir, 'weights.json');
  const opponentPath = join(workdir, 'opponent.json');
  const outputPath = join(workdir, 'trained.json');
  const metricsPath = join(workdir, 'metrics.json');
  const args = [
    '--mode',
    'policy-gradient',
    '--weights',
    weightsPath,
    '--output',
    outputPath,
    '--metrics-output',
    metricsPath,
    '--seed',
    String(options.seed),
    '--matches',
    String(options.matches),
    '--frames',
    String(options.frames),
    '--epochs',
    String(options.epochs),
    '--batch-size',
    String(options.batchSize),
    '--learning-rate',
    String(options.learningRate),
    '--ppo-clip',
    String(options.ppoClip),
    '--temperature',
    String(options.temperature),
    '--discount',
    String(options.discount),
    '--start-state-mode',
    options.startStateMode,
    '--advantage-baseline',
    options.advantageBaseline,
    '--action-mode',
    options.actionMode,
    '--opponent-mode',
    options.opponentMode
  ];

  writeFileSync(weightsPath, JSON.stringify({ weights }), 'utf8');
  if (opponentWeights !== weights) {
    writeFileSync(opponentPath, JSON.stringify({ weights: opponentWeights }), 'utf8');
    args.push('--opponent-weights', opponentPath);
  }
  execFileSync(nativeBin, args, { stdio: 'pipe' });

  const trainedWeights = loadWeightsPayload(readFileSync(outputPath, 'utf8'));
  const metrics = parseNativePolicyGradientMetrics(readFileSync(metricsPath, 'utf8'));
  const finalState = createInitialState();
  finalState.frame = metrics.frames;
  finalState.time = metrics.frames / 30;
  finalState.score = { red: metrics.redGoals, blue: metrics.blueGoals };
  finalState.ball.position = {
    x: metrics.finalBallX,
    y: metrics.finalBallY
  };

  return {
    weights: trainedWeights,
    loss: metrics.loss,
    trainedSamples: metrics.trainedSamples,
    samples: metrics.samples,
    frames: metrics.frames,
    redGoals: metrics.redGoals,
    blueGoals: metrics.blueGoals,
    finalState
  };
}

function resolveNativeTrainer(nativeBin: string | undefined): string {
  if (nativeBin) {
    return nativeBin;
  }

  const defaultPath = join(process.cwd(), 'trainer-rust', 'target', 'release', process.platform === 'win32'
    ? 'soccer-policy-trainer.exe'
    : 'soccer-policy-trainer');
  if (existsSync(defaultPath)) {
    return defaultPath;
  }

  throw new Error('Native trainer not found. Build trainer-rust or pass --rl-native-bin.');
}

function parseNativePolicyGradientMetrics(json: string): {
  samples: number;
  trainedSamples: number;
  frames: number;
  redGoals: number;
  blueGoals: number;
  loss: number;
  finalBallX: number;
  finalBallY: number;
} {
  const parsed = JSON.parse(json) as Record<string, unknown>;
  return {
    samples: finiteMetric(parsed, 'samples'),
    trainedSamples: finiteMetric(parsed, 'trainedSamples'),
    frames: finiteMetric(parsed, 'frames'),
    redGoals: finiteMetric(parsed, 'redGoals'),
    blueGoals: finiteMetric(parsed, 'blueGoals'),
    loss: finiteMetric(parsed, 'loss'),
    finalBallX: finiteMetric(parsed, 'finalBallX'),
    finalBallY: finiteMetric(parsed, 'finalBallY')
  };
}

function finiteMetric(record: Record<string, unknown>, field: string): number {
  const value = record[field];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`Native trainer metrics missing finite ${field}`);
  }
  return value;
}

function createSeededInitialState(seed: number, match: number, team: Team): GameState {
  const random = createSeededRandom(seed + match * 4_099);
  const state = createInitialState();
  const attackFrameX = FIELD.length / 2 + (random() - 0.5) * FIELD.length * 0.16;
  const attackFrameY = FIELD.width / 2 + (random() - 0.5) * FIELD.width * 0.24;

  state.ball.position = fieldPoint(team, attackFrameX, attackFrameY);
  state.ball.velocity = fieldVector(team, (random() - 0.5) * 120, (random() - 0.5) * 120);

  for (const tank of state.tanks) {
    if (tank.team === team) {
      tank.position = fieldPoint(team, 170 + random() * 48, FIELD.width / 2 + (random() - 0.5) * 58);
      tank.angle = team === 'red' ? 0 : Math.PI;
    } else {
      tank.position = fieldPoint(team, FIELD.length - 170 - random() * 48, FIELD.width / 2 + (random() - 0.5) * 58);
      tank.angle = team === 'red' ? Math.PI : 0;
    }
    tank.velocity = { x: 0, y: 0 };
    tank.angularVelocity = 0;
    tank.stamina = tank.maxStamina;
  }

  return state;
}

function attackProgress(state: GameState, initialState: GameState, team: Team): number {
  return (attackX(team, state.ball.position.x) - attackX(team, initialState.ball.position.x)) / FIELD.length;
}

function attackX(team: Team, fieldX: number): number {
  return team === 'red' ? fieldX : FIELD.length - fieldX;
}

function fieldPoint(team: Team, attackFrameX: number, attackFrameY: number): { x: number; y: number } {
  return {
    x: team === 'red' ? attackFrameX : FIELD.length - attackFrameX,
    y: team === 'red' ? attackFrameY : FIELD.width - attackFrameY
  };
}

function fieldVector(team: Team, attackFrameX: number, attackFrameY: number): { x: number; y: number } {
  return {
    x: team === 'red' ? attackFrameX : -attackFrameX,
    y: team === 'red' ? attackFrameY : -attackFrameY
  };
}

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function numberArg(argv: readonly string[], name: string, fallback: number): number {
  const value = valueAfter(argv, name);
  if (value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function positiveIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.max(1, Math.floor(numberArg(argv, name, fallback)));
}

function nonNegativeIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.max(0, Math.floor(numberArg(argv, name, fallback)));
}

function stringArg(argv: readonly string[], name: string): string | undefined {
  return valueAfter(argv, name);
}

function opponentArg(argv: readonly string[], name: string): AcceptOpponentName | undefined {
  const value = valueAfter(argv, name);
  return value === 'idle' ||
    value === 'traditional' ||
    value === 'neural-default' ||
    value === 'neural-self' ||
    value === 'neural-current' ||
    value === 'league' ||
    value === 'runtime'
    ? value
    : undefined;
}

function valueAfter(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index === -1 || index === argv.length - 1) {
    return undefined;
  }
  return argv[index + 1];
}

function startStateModeArg(
  argv: readonly string[],
  name: string,
  fallback: PolicyGradientStartStateMode
): PolicyGradientStartStateMode {
  const value = valueAfter(argv, name);
  return value === 'open' ||
    value === 'outcome-curriculum' ||
    value === 'own-goal-defense' ||
    value === 'corner-fight' ||
    value === 'loose-ball-contest' ||
    value === 'mixed'
    ? value
    : fallback;
}

function advantageBaselineArg(
  argv: readonly string[],
  name: string,
  fallback: PolicyGradientAdvantageBaseline
): PolicyGradientAdvantageBaseline {
  const value = valueAfter(argv, name);
  return value === 'global' || value === 'start-team-time' || value === 'learned'
    ? value
    : fallback;
}

function actionModeArg(
  argv: readonly string[],
  name: string,
  fallback: 'raw' | 'runtime'
): 'raw' | 'runtime' {
  const value = valueAfter(argv, name);
  return value === 'raw' || value === 'runtime'
    ? value
    : fallback;
}

function opponentModeArg(
  argv: readonly string[],
  name: string,
  fallback: 'self' | 'traditional' | 'league'
): 'self' | 'traditional' | 'league' {
  const value = valueAfter(argv, name);
  return value === 'self' || value === 'traditional' || value === 'league'
    ? value
    : fallback;
}

function validateWeights(weights: readonly unknown[]): number[] {
  if (weights.length !== NEURAL_WEIGHT_COUNT) {
    throw new Error(`Expected ${NEURAL_WEIGHT_COUNT} weights, received ${weights.length}`);
  }
  if (!weights.every((weight) => typeof weight === 'number' && Number.isFinite(weight))) {
    throw new Error('Expected every neural weight to be a finite number');
  }
  return [...weights] as number[];
}

function validateReplaySample(sample: unknown, index: number): LearningSample {
  if (!isRecord(sample)) {
    throw new Error(`Expected replay sample ${index} to be an object`);
  }

  const inputs = sample.inputs;
  if (!Array.isArray(inputs) || inputs.length !== POLICY_INPUT_COUNT ||
    !inputs.every((value) => typeof value === 'number' && Number.isFinite(value))) {
    throw new Error(`Expected replay sample inputs to contain ${POLICY_INPUT_COUNT} finite numbers`);
  }

  if (typeof sample.actionIndex !== 'number' ||
    !Number.isFinite(sample.actionIndex) ||
    sample.actionIndex < 0 ||
    sample.actionIndex >= POLICY_ACTION_COUNT) {
    throw new Error(`Expected replay sample ${index} actionIndex to be in policy range`);
  }

  if (sample.team !== 'red' && sample.team !== 'blue') {
    throw new Error(`Expected replay sample ${index} team to be red or blue`);
  }

  if (typeof sample.frame !== 'number' || !Number.isFinite(sample.frame)) {
    throw new Error(`Expected replay sample ${index} frame to be finite`);
  }

  if (!Array.isArray(sample.tags) || !sample.tags.every(isLearningTag)) {
    throw new Error(`Expected replay sample ${index} tags to be valid learning tags`);
  }

  if (typeof sample.weight !== 'number' || !Number.isFinite(sample.weight)) {
    throw new Error(`Expected replay sample ${index} weight to be finite`);
  }

  return {
    inputs: [...inputs],
    actionIndex: Math.round(sample.actionIndex),
    team: sample.team,
    frame: sample.frame,
    tags: [...sample.tags],
    weight: sample.weight
  };
}

function isLearningTag(value: unknown): value is LearningTag {
  return value === 'corner' ||
    value === 'sideWall' ||
    value === 'ownDanger' ||
    value === 'finish' ||
    value === 'lowStamina' ||
    value === 'contact' ||
    value === 'contest';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/coach-neural.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/coach-neural.js')) {
  main();
}
