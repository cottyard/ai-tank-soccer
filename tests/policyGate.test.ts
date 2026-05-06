import { describe, expect, it } from 'vitest';
import {
  compareRuntimeTraces,
  evaluatePolicyGate,
  evaluateRuntimePolicy,
  selectAcceptedPolicy,
  traceRuntimePolicy,
  type RuntimeTraceSummary
} from '../src/ai/policyGate';
import { POLICY_ACTION_COUNT } from '../src/ai/policyActions';
import { parseTraceRuntimePolicyArgs } from '../scripts/trace-runtime-policy';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import type { EvaluationOptions, EvaluationResult } from '../src/ai/neuralTraining';
import type { NeuralWeights } from '../src/ai/neuralWeights';

function scoredWeights(score: number): number[] {
  const weights = defaultNeuralWeights();
  weights[0] = score;
  return weights;
}

describe('policy adoption gate', () => {
  it('accepts a candidate only when its evaluation score improves enough', () => {
    const evaluate = (weights: NeuralWeights, _options: EvaluationOptions): EvaluationResult => ({
      score: weights[0],
      goalDiff: 0,
      ballProgress: 0
    });

    const rejected = evaluatePolicyGate(scoredWeights(10), scoredWeights(10.2), {
      evaluate,
      minDelta: 0.5
    });
    const accepted = evaluatePolicyGate(scoredWeights(10), scoredWeights(10.6), {
      evaluate,
      minDelta: 0.5
    });

    expect(rejected).toMatchObject({
      accepted: false,
      currentScore: 10,
      candidateScore: 10.2
    });
    expect(accepted).toMatchObject({
      accepted: true,
      currentScore: 10,
      candidateScore: 10.6
    });
  });

  it('selects the baseline weights when a saved candidate does not pass the gate', () => {
    const evaluate = (weights: NeuralWeights, _options: EvaluationOptions): EvaluationResult => ({
      score: weights[0],
      goalDiff: 0,
      ballProgress: 0
    });
    const baseline = scoredWeights(20);
    const weakSaved = scoredWeights(18);
    const strongSaved = scoredWeights(22);

    const rejected = selectAcceptedPolicy(baseline, weakSaved, { evaluate });
    const accepted = selectAcceptedPolicy(baseline, strongSaved, { evaluate });

    expect(rejected.source).toBe('current');
    expect(rejected.weights).toEqual(baseline);
    expect(accepted.source).toBe('candidate');
    expect(accepted.weights).toEqual(strongSaved);
  });

  it('can evaluate the runtime tactical strategy against the traditional opponent', () => {
    const result = evaluateRuntimePolicy(defaultNeuralWeights(), {
      seed: 5,
      matches: 2,
      frames: 60
    });

    expect(Number.isFinite(result.score)).toBe(true);
    expect(Number.isFinite(result.goalDiff)).toBe(true);
    expect(Number.isFinite(result.ballProgress)).toBe(true);
    expect(result.goalsFor).toBeGreaterThanOrEqual(0);
    expect(result.goalsAgainst).toBeGreaterThanOrEqual(0);
    expect(result.winProxy).toBeGreaterThanOrEqual(0);
    expect(result.winProxy).toBeLessThanOrEqual(1);
  });

  it('can trace runtime decisions without changing gate scoring', () => {
    const options = {
      seeds: [5, 7],
      matches: 2,
      frames: 60
    };
    const traced = traceRuntimePolicy(defaultNeuralWeights(), options);
    const seed5 = evaluateRuntimePolicy(defaultNeuralWeights(), {
      seed: 5,
      matches: 2,
      frames: 60
    });
    const seed7 = evaluateRuntimePolicy(defaultNeuralWeights(), {
      seed: 7,
      matches: 2,
      frames: 60
    });

    expect(traced.seeds).toHaveLength(2);
    expect(traced.decisions).toBeGreaterThan(0);
    expect(traced.policyActionCounts).toHaveLength(POLICY_ACTION_COUNT);
    expect(traced.tacticalActionCounts).toHaveLength(POLICY_ACTION_COUNT);
    expect(traced.finalActionCounts).toHaveLength(POLICY_ACTION_COUNT);
    expect(sum(traced.finalActionCounts)).toBe(traced.decisions);
    expect(traced.goalsFor).toBe(seed5.goalsFor + seed7.goalsFor);
    expect(traced.goalsAgainst).toBe(seed5.goalsAgainst + seed7.goalsAgainst);
    expect(traced.score).toBeCloseTo((seed5.score + seed7.score) / 2, 9);
    expect(traced.averageStamina).toBeGreaterThanOrEqual(0);
    expect(traced.averageStamina).toBeLessThanOrEqual(1);
  });

  it('parses split seed lists for runtime trace diagnostics', () => {
    const options = parseTraceRuntimePolicyArgs([
      '--seeds',
      '83',
      '97',
      '109',
      '127',
      '149',
      '--matches',
      '4'
    ]);

    expect(options.seeds).toEqual([83, 97, 109, 127, 149]);
    expect(options.matches).toBe(4);
  });

  it('compares runtime traces as behavior-visibility deltas', () => {
    const current = traceSummary({
      decisions: 100,
      finalActionCounts: [50, 50, 0, 0, 0, 0, 0, 0, 0],
      policyActionCounts: [40, 60, 0, 0, 0, 0, 0, 0, 0],
      tacticalActionCounts: [45, 55, 0, 0, 0, 0, 0, 0, 0],
      tacticalRolloutChanges: 20,
      staminaConserves: 10,
      criticalStaminaRegulations: 5
    });
    const candidate = traceSummary({
      decisions: 100,
      finalActionCounts: [40, 55, 5, 0, 0, 0, 0, 0, 0],
      policyActionCounts: [35, 60, 5, 0, 0, 0, 0, 0, 0],
      tacticalActionCounts: [35, 60, 5, 0, 0, 0, 0, 0, 0],
      tacticalRolloutChanges: 30,
      staminaConserves: 8,
      criticalStaminaRegulations: 9
    });

    const delta = compareRuntimeTraces(candidate, current);

    expect(delta.finalActionCounts).toEqual([-10, 5, 5, 0, 0, 0, 0, 0, 0]);
    expect(delta.finalActionDistributionChangeCount).toBe(10);
    expect(delta.finalActionDistributionChangeRate).toBe(0.1);
    expect(delta.policyActionDistributionChangeCount).toBe(5);
    expect(delta.tacticalActionDistributionChangeCount).toBe(10);
    expect(delta.tacticalRolloutChangeRate).toBeCloseTo(0.1, 9);
    expect(delta.staminaConserveRate).toBeCloseTo(-0.02, 9);
    expect(delta.criticalStaminaRegulationRate).toBeCloseTo(0.04, 9);
  });
});

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function traceSummary(overrides: Partial<RuntimeTraceSummary>): RuntimeTraceSummary {
  const actionCounts = Array.from({ length: POLICY_ACTION_COUNT }, () => 0);
  return {
    score: 0,
    goalDiff: 0,
    ballProgress: 0,
    goalsFor: 0,
    goalsAgainst: 0,
    winProxy: 0,
    decisions: 0,
    policyActionCounts: [...actionCounts],
    tacticalActionCounts: [...actionCounts],
    finalActionCounts: [...actionCounts],
    tacticalRolloutUses: 0,
    tacticalRolloutChanges: 0,
    staminaConserves: 0,
    criticalStaminaRegulations: 0,
    flatPolicies: 0,
    averageStamina: 0,
    averageBallDistance: 0,
    averageBallSpeed: 0,
    averageFinishingPressure: 0,
    averageOwnGoalPressure: 0,
    averageSideWallPressure: 0,
    averageAttackCornerPressure: 0,
    averageOwnCornerPressure: 0,
    seeds: [],
    ...overrides
  };
}
