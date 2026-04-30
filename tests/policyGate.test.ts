import { describe, expect, it } from 'vitest';
import { evaluatePolicyGate, evaluateRuntimePolicy, selectAcceptedPolicy } from '../src/ai/policyGate';
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
});
