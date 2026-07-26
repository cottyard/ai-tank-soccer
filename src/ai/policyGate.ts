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
import type { Strategy } from '../game/strategy';

export type PolicyGateOptions = EvaluationOptions & {
  minDelta?: number;
  evaluate?: (weights: NeuralWeights, options: EvaluationOptions) => EvaluationResult;
};

export type RuntimeEvaluationOptions = {
  seed?: number;
  matches?: number;
  frames?: number;
  tacticalRollout?: boolean;
  pairedStarts?: boolean;
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

export type RuntimeTraceDelta = RuntimeEvaluationResult & {
  decisions: number;
  policyActionCounts: number[];
  tacticalActionCounts: number[];
  finalActionCounts: number[];
  policyActionDistributionChangeCount: number;
  policyActionDistributionChangeRate: number;
  tacticalActionDistributionChangeCount: number;
  tacticalActionDistributionChangeRate: number;
  finalActionDistributionChangeCount: number;
  finalActionDistributionChangeRate: number;
  tacticalRolloutUseRate: number;
  tacticalRolloutChangeRate: number;
  staminaConserveRate: number;
  criticalStaminaRegulationRate: number;
  flatPolicyRate: number;
  averageStamina: number;
  averageBallDistance: number;
  averageBallSpeed: number;
  averageFinishingPressure: number;
  averageOwnGoalPressure: number;
  averageSideWallPressure: number;
  averageAttackCornerPressure: number;
  averageOwnCornerPressure: number;
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

export type RuntimeDecisionTraceRecord = NeuralDecisionTrace & {
  seed: number;
  match: number;
  controlledTeam: Team;
  decisionIndex: number;
};

export type RuntimeDecisionTraceRun = {
  summary: RuntimeTraceSummary;
  decisions: RuntimeDecisionTraceRecord[];
};

export type RuntimeDecisionTraceComparison = {
  comparedDecisions: number;
  alignedComparedDecisions: number;
  afterFinalActionDivergenceComparedDecisions: number;
  missingCurrentDecisions: number;
  missingCandidateDecisions: number;
  rawPolicyChanges: number;
  alignedRawPolicyChanges: number;
  afterFinalActionDivergenceRawPolicyChanges: number;
  tacticalActionChanges: number;
  finalActionChanges: number;
  lostPolicyChanges: number;
  alignedLostPolicyChanges: number;
  afterFinalActionDivergenceLostPolicyChanges: number;
  lostWithTacticalRollout: number;
  lostWithStaminaConserve: number;
  lostWithCriticalStamina: number;
  lostWithFlatPolicy: number;
  firstFinalActionDivergences: RuntimeDecisionTraceDivergence[];
  seeds: RuntimeDecisionTraceSeedComparison[];
  samples: RuntimeDecisionLostPolicyChangeSample[];
};

export type RuntimeDecisionTraceSeedComparison = Omit<RuntimeDecisionTraceComparison, 'seeds' | 'samples'> & {
  seed: number;
};

export type RuntimeDecisionTraceDivergence = {
  seed: number;
  match: number;
  controlledTeam: Team;
  decisionIndex: number;
  frame: number;
  currentRawPolicyActionIndex?: number;
  candidateRawPolicyActionIndex?: number;
  currentTacticalActionIndex?: number;
  candidateTacticalActionIndex?: number;
  currentTacticalActionScores?: number[];
  candidateTacticalActionScores?: number[];
  currentFinalActionIndex: number;
  candidateFinalActionIndex: number;
  staminaRatio: number;
  ballDistance: number;
  ballSpeed: number;
  finishingPressure: number;
  ownGoalPressure: number;
  sideWallPressure: number;
  attackCornerPressure: number;
  ownCornerPressure: number;
};

export type RuntimeDecisionLostPolicyChangeSample = {
  seed: number;
  match: number;
  controlledTeam: Team;
  decisionIndex: number;
  frame: number;
  currentRawPolicyActionIndex?: number;
  candidateRawPolicyActionIndex?: number;
  currentTacticalActionIndex?: number;
  candidateTacticalActionIndex?: number;
  currentTacticalActionScores?: number[];
  candidateTacticalActionScores?: number[];
  finalActionIndex: number;
  reasons: string[];
  staminaRatio: number;
  ballDistance: number;
  ballSpeed: number;
  finishingPressure: number;
  ownGoalPressure: number;
  sideWallPressure: number;
  attackCornerPressure: number;
  ownCornerPressure: number;
  afterFinalActionDivergence: boolean;
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

export function evaluateRuntimePolicyAgainst(
  weights: NeuralWeights,
  opponent: Strategy,
  options: RuntimeEvaluationOptions = {}
): RuntimeEvaluationResult {
  return evaluateRuntimePolicyInternal(weights, options, false, false, opponent).result;
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

  return traceSummaryFromTotals(totals, seeds.length, seedSummaries);
}

export function traceRuntimePolicyDecisions(
  weights: NeuralWeights,
  options: RuntimeEvaluationOptions & { seeds?: readonly number[] } = {}
): RuntimeDecisionTraceRun {
  const seeds = options.seeds && options.seeds.length > 0
    ? options.seeds
    : [options.seed ?? 1];
  const totals = emptyTraceTotals();
  const decisions: RuntimeDecisionTraceRecord[] = [];
  const seedSummaries = seeds.map((seed) => {
    const traced = evaluateRuntimePolicyInternal(weights, {
      ...options,
      seed
    }, true, true);
    accumulateTraceTotals(totals, traced);
    decisions.push(...traced.decisionRecords);
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
    summary: traceSummaryFromTotals(totals, seeds.length, seedSummaries),
    decisions
  };
}

export function compareRuntimeTraces(
  candidate: RuntimeTraceSummary,
  current: RuntimeTraceSummary
): RuntimeTraceDelta {
  const policyActionCounts = subtractCounts(candidate.policyActionCounts, current.policyActionCounts);
  const tacticalActionCounts = subtractCounts(candidate.tacticalActionCounts, current.tacticalActionCounts);
  const finalActionCounts = subtractCounts(candidate.finalActionCounts, current.finalActionCounts);
  const policyActionDistributionChangeCount = distributionChangeCount(candidate.policyActionCounts, current.policyActionCounts);
  const tacticalActionDistributionChangeCount = distributionChangeCount(candidate.tacticalActionCounts, current.tacticalActionCounts);
  const finalActionDistributionChangeCount = distributionChangeCount(candidate.finalActionCounts, current.finalActionCounts);
  const actionChangeDenominator = Math.max(candidate.decisions, current.decisions, 1);

  return {
    score: candidate.score - current.score,
    goalDiff: candidate.goalDiff - current.goalDiff,
    ballProgress: candidate.ballProgress - current.ballProgress,
    goalsFor: candidate.goalsFor - current.goalsFor,
    goalsAgainst: candidate.goalsAgainst - current.goalsAgainst,
    winProxy: candidate.winProxy - current.winProxy,
    decisions: candidate.decisions - current.decisions,
    policyActionCounts,
    tacticalActionCounts,
    finalActionCounts,
    policyActionDistributionChangeCount,
    policyActionDistributionChangeRate: policyActionDistributionChangeCount / actionChangeDenominator,
    tacticalActionDistributionChangeCount,
    tacticalActionDistributionChangeRate: tacticalActionDistributionChangeCount / actionChangeDenominator,
    finalActionDistributionChangeCount,
    finalActionDistributionChangeRate: finalActionDistributionChangeCount / actionChangeDenominator,
    tacticalRolloutUseRate: rate(candidate.tacticalRolloutUses, candidate.decisions) -
      rate(current.tacticalRolloutUses, current.decisions),
    tacticalRolloutChangeRate: rate(candidate.tacticalRolloutChanges, candidate.decisions) -
      rate(current.tacticalRolloutChanges, current.decisions),
    staminaConserveRate: rate(candidate.staminaConserves, candidate.decisions) -
      rate(current.staminaConserves, current.decisions),
    criticalStaminaRegulationRate: rate(candidate.criticalStaminaRegulations, candidate.decisions) -
      rate(current.criticalStaminaRegulations, current.decisions),
    flatPolicyRate: rate(candidate.flatPolicies, candidate.decisions) -
      rate(current.flatPolicies, current.decisions),
    averageStamina: candidate.averageStamina - current.averageStamina,
    averageBallDistance: candidate.averageBallDistance - current.averageBallDistance,
    averageBallSpeed: candidate.averageBallSpeed - current.averageBallSpeed,
    averageFinishingPressure: candidate.averageFinishingPressure - current.averageFinishingPressure,
    averageOwnGoalPressure: candidate.averageOwnGoalPressure - current.averageOwnGoalPressure,
    averageSideWallPressure: candidate.averageSideWallPressure - current.averageSideWallPressure,
    averageAttackCornerPressure: candidate.averageAttackCornerPressure - current.averageAttackCornerPressure,
    averageOwnCornerPressure: candidate.averageOwnCornerPressure - current.averageOwnCornerPressure
  };
}

export function compareRuntimeDecisionTraces(
  candidate: readonly RuntimeDecisionTraceRecord[],
  current: readonly RuntimeDecisionTraceRecord[],
  sampleLimit = 20
): RuntimeDecisionTraceComparison {
  const currentByKey = decisionRecordsByKey(current);
  const candidateByKey = decisionRecordsByKey(candidate);
  const seedTotals = new Map<number, MutableDecisionTraceComparison>();
  const totals = emptyDecisionTraceComparison();
  const samples: RuntimeDecisionLostPolicyChangeSample[] = [];
  const divergenceByMatch = new Map<string, RuntimeDecisionTraceDivergence>();
  const keys = new Set([...currentByKey.keys(), ...candidateByKey.keys()]);

  for (const key of [...keys].sort(compareDecisionKeys)) {
    const currentRecord = currentByKey.get(key);
    const candidateRecord = candidateByKey.get(key);
    if (!currentRecord) {
      totals.missingCurrentDecisions += 1;
      seedMutableTotals(seedTotals, candidateRecord!.seed).missingCurrentDecisions += 1;
      continue;
    }
    if (!candidateRecord) {
      totals.missingCandidateDecisions += 1;
      seedMutableTotals(seedTotals, currentRecord.seed).missingCandidateDecisions += 1;
      continue;
    }

    const seed = seedMutableTotals(seedTotals, currentRecord.seed);
    const matchKey = decisionMatchKey(currentRecord);
    const afterFinalActionDivergence = divergenceByMatch.has(matchKey);
    if (!afterFinalActionDivergence && candidateRecord.finalActionIndex !== currentRecord.finalActionIndex) {
      const divergence = {
        seed: currentRecord.seed,
        match: currentRecord.match,
        controlledTeam: currentRecord.controlledTeam,
        decisionIndex: currentRecord.decisionIndex,
        frame: candidateRecord.frame,
        currentRawPolicyActionIndex: currentRecord.rawPolicyActionIndex,
        candidateRawPolicyActionIndex: candidateRecord.rawPolicyActionIndex,
        currentTacticalActionIndex: currentRecord.tacticalActionIndex,
        candidateTacticalActionIndex: candidateRecord.tacticalActionIndex,
        currentTacticalActionScores: currentRecord.tacticalActionScores ? [...currentRecord.tacticalActionScores] : undefined,
        candidateTacticalActionScores: candidateRecord.tacticalActionScores ? [...candidateRecord.tacticalActionScores] : undefined,
        currentFinalActionIndex: currentRecord.finalActionIndex,
        candidateFinalActionIndex: candidateRecord.finalActionIndex,
        staminaRatio: candidateRecord.staminaRatio,
        ballDistance: candidateRecord.ballDistance,
        ballSpeed: candidateRecord.ballSpeed,
        finishingPressure: candidateRecord.finishingPressure,
        ownGoalPressure: candidateRecord.ownGoalPressure,
        sideWallPressure: candidateRecord.sideWallPressure,
        attackCornerPressure: candidateRecord.attackCornerPressure,
        ownCornerPressure: candidateRecord.ownCornerPressure
      };
      divergenceByMatch.set(matchKey, divergence);
      totals.firstFinalActionDivergences.push(divergence);
      seed.firstFinalActionDivergences.push(divergence);
    }
    accumulateDecisionComparison(totals, candidateRecord, currentRecord, samples, sampleLimit, afterFinalActionDivergence);
    accumulateDecisionComparison(seed, candidateRecord, currentRecord, samples, 0, afterFinalActionDivergence);
  }

  return {
    ...totals,
    seeds: [...seedTotals.entries()]
      .sort(([a], [b]) => a - b)
      .map(([seed, row]) => ({ seed, ...row })),
    samples
  };
}

function evaluateRuntimePolicyInternal(
  weights: NeuralWeights,
  options: RuntimeEvaluationOptions = {},
  collectTrace: boolean,
  collectDecisionRecords = false,
  opponentStrategy: Strategy = traditionalStrategy
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
  decisionRecords: RuntimeDecisionTraceRecord[];
} {
  const seed = options.seed ?? 1;
  const matches = Math.max(1, Math.floor(options.matches ?? 8));
  const frames = Math.max(1, Math.floor(options.frames ?? 30 * 30));
  const traceTotals = emptyTraceTotals();
  const decisionRecords: RuntimeDecisionTraceRecord[] = [];
  let currentMatch = 0;
  let currentTeam: Team = 'red';
  let decisionIndex = 0;
  const neural = createNeuralStrategy({
    weights,
    name: 'neural-runtime-gate',
    tacticalRollout: options.tacticalRollout ?? true,
    onDecision: collectTrace ? (trace) => {
      recordDecisionTrace(traceTotals, trace);
      if (collectDecisionRecords) {
        decisionRecords.push({
          seed,
          match: currentMatch,
          controlledTeam: currentTeam,
          decisionIndex,
          ...trace
        });
      }
      decisionIndex += 1;
    } : undefined
  });
  let goalsFor = 0;
  let goalsAgainst = 0;
  let wins = 0;
  let ballProgress = 0;

  for (let match = 0; match < matches; match += 1) {
    const team: Team = match % 2 === 0 ? 'red' : 'blue';
    currentMatch = match;
    currentTeam = team;
    decisionIndex = 0;
    const scenario = options.pairedStarts ? Math.floor(match / 2) : match;
    const initialState = createSeededInitialState(
      seed,
      scenario,
      options.pairedStarts ? 'red' : team
    );
    const result = simulateMatch({
      red: team === 'red' ? neural : opponentStrategy,
      blue: team === 'blue' ? neural : opponentStrategy,
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
    ownCornerPressureSum: traceTotals.ownCornerPressureSum,
    decisionRecords
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

function traceSummaryFromTotals(
  totals: RuntimeTraceTotals,
  seedCount: number,
  seedSummaries: RuntimeTraceSeedSummary[]
): RuntimeTraceSummary {
  const divisor = Math.max(1, seedCount);
  return {
    score: totals.score / divisor,
    goalDiff: totals.goalsFor - totals.goalsAgainst,
    ballProgress: totals.ballProgress / divisor,
    goalsFor: totals.goalsFor,
    goalsAgainst: totals.goalsAgainst,
    winProxy: totals.winProxy / divisor,
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

type MutableDecisionTraceComparison = Omit<RuntimeDecisionTraceComparison, 'seeds' | 'samples'>;

function emptyDecisionTraceComparison(): MutableDecisionTraceComparison {
  return {
    comparedDecisions: 0,
    alignedComparedDecisions: 0,
    afterFinalActionDivergenceComparedDecisions: 0,
    missingCurrentDecisions: 0,
    missingCandidateDecisions: 0,
    rawPolicyChanges: 0,
    alignedRawPolicyChanges: 0,
    afterFinalActionDivergenceRawPolicyChanges: 0,
    tacticalActionChanges: 0,
    finalActionChanges: 0,
    lostPolicyChanges: 0,
    alignedLostPolicyChanges: 0,
    afterFinalActionDivergenceLostPolicyChanges: 0,
    lostWithTacticalRollout: 0,
    lostWithStaminaConserve: 0,
    lostWithCriticalStamina: 0,
    lostWithFlatPolicy: 0,
    firstFinalActionDivergences: []
  };
}

function seedMutableTotals(
  totals: Map<number, MutableDecisionTraceComparison>,
  seed: number
): MutableDecisionTraceComparison {
  const existing = totals.get(seed);
  if (existing) {
    return existing;
  }
  const created = emptyDecisionTraceComparison();
  totals.set(seed, created);
  return created;
}

function accumulateDecisionComparison(
  totals: MutableDecisionTraceComparison,
  candidate: RuntimeDecisionTraceRecord,
  current: RuntimeDecisionTraceRecord,
  samples: RuntimeDecisionLostPolicyChangeSample[],
  sampleLimit: number,
  afterFinalActionDivergence: boolean
): void {
  totals.comparedDecisions += 1;
  totals.alignedComparedDecisions += afterFinalActionDivergence ? 0 : 1;
  totals.afterFinalActionDivergenceComparedDecisions += afterFinalActionDivergence ? 1 : 0;
  const rawPolicyChanged = candidate.rawPolicyActionIndex !== current.rawPolicyActionIndex;
  const tacticalActionChanged = candidate.tacticalActionIndex !== current.tacticalActionIndex;
  const finalActionChanged = candidate.finalActionIndex !== current.finalActionIndex;

  totals.rawPolicyChanges += rawPolicyChanged ? 1 : 0;
  totals.alignedRawPolicyChanges += rawPolicyChanged && !afterFinalActionDivergence ? 1 : 0;
  totals.afterFinalActionDivergenceRawPolicyChanges += rawPolicyChanged && afterFinalActionDivergence ? 1 : 0;
  totals.tacticalActionChanges += tacticalActionChanged ? 1 : 0;
  totals.finalActionChanges += finalActionChanged ? 1 : 0;

  if (!rawPolicyChanged || finalActionChanged) {
    return;
  }

  const reasons = hiddenPolicyChangeReasons(candidate, current);
  totals.lostPolicyChanges += 1;
  totals.alignedLostPolicyChanges += afterFinalActionDivergence ? 0 : 1;
  totals.afterFinalActionDivergenceLostPolicyChanges += afterFinalActionDivergence ? 1 : 0;
  totals.lostWithTacticalRollout += reasons.includes('tactical-rollout') ? 1 : 0;
  totals.lostWithStaminaConserve += reasons.includes('stamina-conserve') ? 1 : 0;
  totals.lostWithCriticalStamina += reasons.includes('critical-stamina') ? 1 : 0;
  totals.lostWithFlatPolicy += reasons.includes('flat-policy') ? 1 : 0;

  if (sampleLimit > 0 && samples.length < sampleLimit) {
    samples.push({
      seed: candidate.seed,
      match: candidate.match,
      controlledTeam: candidate.controlledTeam,
      decisionIndex: candidate.decisionIndex,
      frame: candidate.frame,
      currentRawPolicyActionIndex: current.rawPolicyActionIndex,
      candidateRawPolicyActionIndex: candidate.rawPolicyActionIndex,
      currentTacticalActionIndex: current.tacticalActionIndex,
      candidateTacticalActionIndex: candidate.tacticalActionIndex,
      currentTacticalActionScores: current.tacticalActionScores ? [...current.tacticalActionScores] : undefined,
      candidateTacticalActionScores: candidate.tacticalActionScores ? [...candidate.tacticalActionScores] : undefined,
      finalActionIndex: candidate.finalActionIndex,
      reasons,
      staminaRatio: candidate.staminaRatio,
      ballDistance: candidate.ballDistance,
      ballSpeed: candidate.ballSpeed,
      finishingPressure: candidate.finishingPressure,
      ownGoalPressure: candidate.ownGoalPressure,
      sideWallPressure: candidate.sideWallPressure,
      attackCornerPressure: candidate.attackCornerPressure,
      ownCornerPressure: candidate.ownCornerPressure,
      afterFinalActionDivergence
    });
  }
}

function hiddenPolicyChangeReasons(
  candidate: RuntimeDecisionTraceRecord,
  current: RuntimeDecisionTraceRecord
): string[] {
  const reasons: string[] = [];
  if (candidate.tacticalRolloutChanged || current.tacticalRolloutChanged) {
    reasons.push('tactical-rollout');
  }
  if (candidate.staminaConserved || current.staminaConserved) {
    reasons.push('stamina-conserve');
  }
  if (candidate.criticalStaminaRegulated || current.criticalStaminaRegulated) {
    reasons.push('critical-stamina');
  }
  if (candidate.flatPolicy || current.flatPolicy) {
    reasons.push('flat-policy');
  }
  if (reasons.length === 0) {
    reasons.push('unchanged-final-action');
  }
  return reasons;
}

function decisionRecordsByKey(
  records: readonly RuntimeDecisionTraceRecord[]
): Map<string, RuntimeDecisionTraceRecord> {
  const byKey = new Map<string, RuntimeDecisionTraceRecord>();
  for (const record of records) {
    byKey.set(decisionRecordKey(record), record);
  }
  return byKey;
}

function decisionRecordKey(record: RuntimeDecisionTraceRecord): string {
  return `${record.seed}|${record.match}|${record.controlledTeam}|${record.decisionIndex}`;
}

function decisionMatchKey(record: RuntimeDecisionTraceRecord): string {
  return `${record.seed}|${record.match}|${record.controlledTeam}`;
}

function compareDecisionKeys(a: string, b: string): number {
  const [aSeed, aMatch, aTeam, aDecision] = a.split('|');
  const [bSeed, bMatch, bTeam, bDecision] = b.split('|');
  return Number(aSeed) - Number(bSeed) ||
    Number(aMatch) - Number(bMatch) ||
    aTeam.localeCompare(bTeam) ||
    Number(aDecision) - Number(bDecision);
}

function safeAverage(total: number, count: number): number {
  return count > 0 ? total / count : 0;
}

function rate(count: number, total: number): number {
  return total > 0 ? count / total : 0;
}

function subtractCounts(candidate: readonly number[], current: readonly number[]): number[] {
  const length = Math.max(candidate.length, current.length);
  return Array.from({ length }, (_, index) => (candidate[index] ?? 0) - (current[index] ?? 0));
}

function distributionChangeCount(candidate: readonly number[], current: readonly number[]): number {
  return subtractCounts(candidate, current)
    .reduce((total, count) => total + Math.abs(count), 0) / 2;
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
