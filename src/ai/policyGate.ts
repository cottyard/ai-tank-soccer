import {
  evaluateNeuralWeights,
  type EvaluationOptions,
  type EvaluationResult
} from './neuralTraining';
import { createNeuralStrategy, type NeuralDecisionTrace } from './neuralStrategy';
import type { NeuralWeights } from './neuralWeights';
import { POLICY_ACTION_COUNT } from './policyActions';
import { traditionalStrategy } from './traditionalStrategy';
import { FIELD, createInitialState, type GameState, type Team } from '../game/model';
import { simulateMatch } from '../game/match';

export type PolicyGateOptions = EvaluationOptions & {
  minDelta?: number;
  evaluate?: (weights: NeuralWeights, options: EvaluationOptions) => EvaluationResult;
};

export type RuntimeEvaluationOptions = {
  seed?: number;
  matches?: number;
  frames?: number;
};

export type RuntimeEvaluationResult = EvaluationResult & {
  goalsFor: number;
  goalsAgainst: number;
  winProxy: number;
};

export type RuntimeTraceSummary = RuntimeEvaluationResult & {
  decisions: number;
  policyActionCounts: number[];
  tacticalActionCounts: number[];
  finalActionCounts: number[];
  tacticalRolloutUses: number;
  tacticalRolloutChanges: number;
  staminaConserves: number;
  criticalStaminaRegulations: number;
  flatPolicies: number;
  averageStamina: number;
  averageBallDistance: number;
  averageBallSpeed: number;
  averageFinishingPressure: number;
  averageOwnGoalPressure: number;
  averageSideWallPressure: number;
  averageAttackCornerPressure: number;
  averageOwnCornerPressure: number;
  seeds: RuntimeTraceSeedSummary[];
};

export type RuntimeTraceSeedSummary = RuntimeEvaluationResult & {
  seed: number;
  decisions: number;
  tacticalRolloutUses: number;
  tacticalRolloutChanges: number;
  staminaConserves: number;
  criticalStaminaRegulations: number;
  flatPolicies: number;
};

export type PolicyGateResult = {
  accepted: boolean;
  currentScore: number;
  candidateScore: number;
  current: EvaluationResult;
  candidate: EvaluationResult;
};

export type SelectedPolicyResult = PolicyGateResult & {
  weights: number[];
  source: 'current' | 'candidate';
};

export function evaluatePolicyGate(
  currentWeights: NeuralWeights,
  candidateWeights: NeuralWeights,
  options: PolicyGateOptions = {}
): PolicyGateResult {
  const { evaluate = evaluateNeuralWeights, minDelta = 0, ...evaluationOptions } = options;
  const current = evaluate(currentWeights, evaluationOptions);
  const candidate = evaluate(candidateWeights, evaluationOptions);

  return {
    accepted: candidate.score > current.score + minDelta,
    currentScore: current.score,
    candidateScore: candidate.score,
    current,
    candidate
  };
}

export function selectAcceptedPolicy(
  currentWeights: NeuralWeights,
  candidateWeights: NeuralWeights,
  options: PolicyGateOptions = {}
): SelectedPolicyResult {
  const gate = evaluatePolicyGate(currentWeights, candidateWeights, options);
  return {
    ...gate,
    weights: gate.accepted ? [...candidateWeights] : [...currentWeights],
    source: gate.accepted ? 'candidate' : 'current'
  };
}

export function evaluateRuntimePolicy(
  weights: NeuralWeights,
  options: RuntimeEvaluationOptions = {}
): RuntimeEvaluationResult {
  return evaluateRuntimePolicyInternal(weights, options, false).result;
}

export function traceRuntimePolicy(
  weights: NeuralWeights,
  options: RuntimeEvaluationOptions & { seeds?: readonly number[] } = {}
): RuntimeTraceSummary {
  const seeds = options.seeds && options.seeds.length > 0
    ? options.seeds
    : [options.seed ?? 1];
  const totals = emptyTraceTotals();
  const seedSummaries = seeds.map((seed) => {
    const traced = evaluateRuntimePolicyInternal(weights, {
      ...options,
      seed
    }, true);
    accumulateTraceTotals(totals, traced);
    return {
      seed,
      ...traced.result,
      decisions: traced.decisions,
      tacticalRolloutUses: traced.tacticalRolloutUses,
      tacticalRolloutChanges: traced.tacticalRolloutChanges,
      staminaConserves: traced.staminaConserves,
      criticalStaminaRegulations: traced.criticalStaminaRegulations,
      flatPolicies: traced.flatPolicies
    };
  });

  return {
    score: totals.score / seeds.length,
    goalDiff: totals.goalsFor - totals.goalsAgainst,
    ballProgress: totals.ballProgress / seeds.length,
    goalsFor: totals.goalsFor,
    goalsAgainst: totals.goalsAgainst,
    winProxy: totals.winProxy / seeds.length,
    decisions: totals.decisions,
    policyActionCounts: totals.policyActionCounts,
    tacticalActionCounts: totals.tacticalActionCounts,
    finalActionCounts: totals.finalActionCounts,
    tacticalRolloutUses: totals.tacticalRolloutUses,
    tacticalRolloutChanges: totals.tacticalRolloutChanges,
    staminaConserves: totals.staminaConserves,
    criticalStaminaRegulations: totals.criticalStaminaRegulations,
    flatPolicies: totals.flatPolicies,
    averageStamina: safeAverage(totals.staminaSum, totals.decisions),
    averageBallDistance: safeAverage(totals.ballDistanceSum, totals.decisions),
    averageBallSpeed: safeAverage(totals.ballSpeedSum, totals.decisions),
    averageFinishingPressure: safeAverage(totals.finishingPressureSum, totals.decisions),
    averageOwnGoalPressure: safeAverage(totals.ownGoalPressureSum, totals.decisions),
    averageSideWallPressure: safeAverage(totals.sideWallPressureSum, totals.decisions),
    averageAttackCornerPressure: safeAverage(totals.attackCornerPressureSum, totals.decisions),
    averageOwnCornerPressure: safeAverage(totals.ownCornerPressureSum, totals.decisions),
    seeds: seedSummaries
  };
}

function evaluateRuntimePolicyInternal(
  weights: NeuralWeights,
  options: RuntimeEvaluationOptions = {},
  collectTrace: boolean
): {
  result: RuntimeEvaluationResult;
  decisions: number;
  policyActionCounts: number[];
  tacticalActionCounts: number[];
  finalActionCounts: number[];
  tacticalRolloutUses: number;
  tacticalRolloutChanges: number;
  staminaConserves: number;
  criticalStaminaRegulations: number;
  flatPolicies: number;
  staminaSum: number;
  ballDistanceSum: number;
  ballSpeedSum: number;
  finishingPressureSum: number;
  ownGoalPressureSum: number;
  sideWallPressureSum: number;
  attackCornerPressureSum: number;
  ownCornerPressureSum: number;
} {
  const seed = options.seed ?? 1;
  const matches = Math.max(1, Math.floor(options.matches ?? 8));
  const frames = Math.max(1, Math.floor(options.frames ?? 30 * 30));
  const traceTotals = emptyTraceTotals();
  const neural = createNeuralStrategy({
    weights,
    name: 'neural-runtime-gate',
    tacticalRollout: true,
    onDecision: collectTrace ? (trace) => recordDecisionTrace(traceTotals, trace) : undefined
  });
  let goalsFor = 0;
  let goalsAgainst = 0;
  let wins = 0;
  let ballProgress = 0;

  for (let match = 0; match < matches; match += 1) {
    const team: Team = match % 2 === 0 ? 'red' : 'blue';
    const initialState = createSeededInitialState(seed, match, team);
    const result = simulateMatch({
      red: team === 'red' ? neural : traditionalStrategy,
      blue: team === 'blue' ? neural : traditionalStrategy,
      frames,
      initialState
    }).state;
    const forGoals = team === 'red' ? result.score.red : result.score.blue;
    const againstGoals = team === 'red' ? result.score.blue : result.score.red;

    goalsFor += forGoals;
    goalsAgainst += againstGoals;
    wins += forGoals > againstGoals ? 1 : forGoals === againstGoals ? 0.5 : 0;
    ballProgress += attackProgress(result, initialState, team);
  }

  const goalDiff = goalsFor - goalsAgainst;
  const averageBallProgress = ballProgress / matches;
  const winProxy = wins / matches;
  return {
    result: {
      score: goalDiff * 100 + goalsFor * 35 - goalsAgainst * 55 + averageBallProgress * 80 + winProxy * 45,
      goalDiff,
      ballProgress: averageBallProgress,
      goalsFor,
      goalsAgainst,
      winProxy
    },
    decisions: traceTotals.decisions,
    policyActionCounts: traceTotals.policyActionCounts,
    tacticalActionCounts: traceTotals.tacticalActionCounts,
    finalActionCounts: traceTotals.finalActionCounts,
    tacticalRolloutUses: traceTotals.tacticalRolloutUses,
    tacticalRolloutChanges: traceTotals.tacticalRolloutChanges,
    staminaConserves: traceTotals.staminaConserves,
    criticalStaminaRegulations: traceTotals.criticalStaminaRegulations,
    flatPolicies: traceTotals.flatPolicies,
    staminaSum: traceTotals.staminaSum,
    ballDistanceSum: traceTotals.ballDistanceSum,
    ballSpeedSum: traceTotals.ballSpeedSum,
    finishingPressureSum: traceTotals.finishingPressureSum,
    ownGoalPressureSum: traceTotals.ownGoalPressureSum,
    sideWallPressureSum: traceTotals.sideWallPressureSum,
    attackCornerPressureSum: traceTotals.attackCornerPressureSum,
    ownCornerPressureSum: traceTotals.ownCornerPressureSum
  };
}

type RuntimeTraceTotals = {
  score: number;
  ballProgress: number;
  goalsFor: number;
  goalsAgainst: number;
  winProxy: number;
  decisions: number;
  policyActionCounts: number[];
  tacticalActionCounts: number[];
  finalActionCounts: number[];
  tacticalRolloutUses: number;
  tacticalRolloutChanges: number;
  staminaConserves: number;
  criticalStaminaRegulations: number;
  flatPolicies: number;
  staminaSum: number;
  ballDistanceSum: number;
  ballSpeedSum: number;
  finishingPressureSum: number;
  ownGoalPressureSum: number;
  sideWallPressureSum: number;
  attackCornerPressureSum: number;
  ownCornerPressureSum: number;
};

function emptyTraceTotals(): RuntimeTraceTotals {
  return {
    score: 0,
    ballProgress: 0,
    goalsFor: 0,
    goalsAgainst: 0,
    winProxy: 0,
    decisions: 0,
    policyActionCounts: Array.from({ length: POLICY_ACTION_COUNT }, () => 0),
    tacticalActionCounts: Array.from({ length: POLICY_ACTION_COUNT }, () => 0),
    finalActionCounts: Array.from({ length: POLICY_ACTION_COUNT }, () => 0),
    tacticalRolloutUses: 0,
    tacticalRolloutChanges: 0,
    staminaConserves: 0,
    criticalStaminaRegulations: 0,
    flatPolicies: 0,
    staminaSum: 0,
    ballDistanceSum: 0,
    ballSpeedSum: 0,
    finishingPressureSum: 0,
    ownGoalPressureSum: 0,
    sideWallPressureSum: 0,
    attackCornerPressureSum: 0,
    ownCornerPressureSum: 0
  };
}

function accumulateTraceTotals(
  totals: RuntimeTraceTotals,
  traced: ReturnType<typeof evaluateRuntimePolicyInternal>
): void {
  totals.score += traced.result.score;
  totals.ballProgress += traced.result.ballProgress;
  totals.goalsFor += traced.result.goalsFor;
  totals.goalsAgainst += traced.result.goalsAgainst;
  totals.winProxy += traced.result.winProxy;
  totals.decisions += traced.decisions;
  totals.tacticalRolloutUses += traced.tacticalRolloutUses;
  totals.tacticalRolloutChanges += traced.tacticalRolloutChanges;
  totals.staminaConserves += traced.staminaConserves;
  totals.criticalStaminaRegulations += traced.criticalStaminaRegulations;
  totals.flatPolicies += traced.flatPolicies;
  totals.staminaSum += traced.staminaSum;
  totals.ballDistanceSum += traced.ballDistanceSum;
  totals.ballSpeedSum += traced.ballSpeedSum;
  totals.finishingPressureSum += traced.finishingPressureSum;
  totals.ownGoalPressureSum += traced.ownGoalPressureSum;
  totals.sideWallPressureSum += traced.sideWallPressureSum;
  totals.attackCornerPressureSum += traced.attackCornerPressureSum;
  totals.ownCornerPressureSum += traced.ownCornerPressureSum;
  for (let index = 0; index < totals.policyActionCounts.length; index += 1) {
    totals.policyActionCounts[index] += traced.policyActionCounts[index] ?? 0;
    totals.tacticalActionCounts[index] += traced.tacticalActionCounts[index] ?? 0;
    totals.finalActionCounts[index] += traced.finalActionCounts[index] ?? 0;
  }
}

function recordDecisionTrace(totals: RuntimeTraceTotals, trace: NeuralDecisionTrace): void {
  totals.decisions += 1;
  if (trace.policyActionIndex !== undefined) {
    totals.policyActionCounts[trace.policyActionIndex] += 1;
  }
  if (trace.tacticalActionIndex !== undefined) {
    totals.tacticalActionCounts[trace.tacticalActionIndex] += 1;
  }
  totals.finalActionCounts[trace.finalActionIndex] += 1;
  totals.tacticalRolloutUses += trace.tacticalRolloutUsed ? 1 : 0;
  totals.tacticalRolloutChanges += trace.tacticalRolloutChanged ? 1 : 0;
  totals.staminaConserves += trace.staminaConserved ? 1 : 0;
  totals.criticalStaminaRegulations += trace.criticalStaminaRegulated ? 1 : 0;
  totals.flatPolicies += trace.flatPolicy ? 1 : 0;
  totals.staminaSum += trace.staminaRatio;
  totals.ballDistanceSum += trace.ballDistance;
  totals.ballSpeedSum += trace.ballSpeed;
  totals.finishingPressureSum += trace.finishingPressure;
  totals.ownGoalPressureSum += trace.ownGoalPressure;
  totals.sideWallPressureSum += trace.sideWallPressure;
  totals.attackCornerPressureSum += trace.attackCornerPressure;
  totals.ownCornerPressureSum += trace.ownCornerPressure;
}

function safeAverage(total: number, count: number): number {
  return count > 0 ? total / count : 0;
}

function createSeededInitialState(seed: number, match: number, team: Team): GameState {
  const random = createSeededRandom(seed + match * 4099);
  const state = createInitialState();
  const attackFrameX = FIELD.length / 2 + (random() - 0.5) * FIELD.length * 0.12;
  const attackFrameY = FIELD.width / 2 + (random() - 0.5) * FIELD.width * 0.22;

  state.ball.position = fieldPoint(team, attackFrameX, attackFrameY);
  state.ball.velocity = fieldVector(team, (random() - 0.5) * 120, (random() - 0.5) * 120);
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
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
