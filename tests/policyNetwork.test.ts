import { describe, expect, it } from 'vitest';
import { POLICY_ACTION_COUNT } from '../src/ai/policyActions';
import {
  POLICY_HIDDEN_LAYER_SIZES,
  POLICY_HIDDEN_COUNT,
  POLICY_INPUT_COUNT,
  POLICY_OUTPUT_COUNT,
  POLICY_WEIGHT_COUNT,
  createPolicyWeights,
  evaluatePolicy,
  policyProbabilities,
  softmax,
  trainPolicyBatch
  , trainPolicyGradientBatch
} from '../src/ai/policyNetwork';

describe('policy network', () => {
  it('uses the existing neural input size and nine action outputs', () => {
    expect(POLICY_INPUT_COUNT).toBe(36);
    expect(POLICY_OUTPUT_COUNT).toBe(POLICY_ACTION_COUNT);
    expect(POLICY_HIDDEN_LAYER_SIZES).toEqual([64, 64]);
    expect(POLICY_HIDDEN_COUNT).toBe(64);
    expect(POLICY_WEIGHT_COUNT).toBe(7113);
    expect(createPolicyWeights()).toHaveLength(POLICY_WEIGHT_COUNT);
  });

  it('normalizes logits with a stable softmax', () => {
    const probabilities = softmax([1000, 1001, 999]);

    expect(probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 6);
    expect(probabilities[1]).toBeGreaterThan(probabilities[0]);
    expect(probabilities[0]).toBeGreaterThan(probabilities[2]);
  });

  it('increases demonstrated action probability after a supervised update', () => {
    const inputs = Array.from({ length: POLICY_INPUT_COUNT }, (_, index) =>
      index % 3 === 0 ? 0.75 : index % 3 === 1 ? -0.35 : 0.12
    );
    const weights = createPolicyWeights(7);
    const targetAction = 8;
    const before = policyProbabilities(evaluatePolicy(inputs, weights))[targetAction];

    const result = trainPolicyBatch(weights, [{
      inputs,
      actionIndex: targetAction,
      weight: 1
    }], {
      learningRate: 0.18,
      l2: 0,
      gradientClip: 3
    });

    const after = policyProbabilities(evaluatePolicy(inputs, result.weights))[targetAction];

    expect(result.loss).toBeGreaterThan(0);
    expect(after).toBeGreaterThan(before);
  });

  it('increases positive-advantage action probability after a policy-gradient update', () => {
    const inputs = Array.from({ length: POLICY_INPUT_COUNT }, (_, index) =>
      index % 4 === 0 ? 0.5 : index % 4 === 1 ? -0.2 : index % 4 === 2 ? 0.35 : -0.1
    );
    const weights = createPolicyWeights(17);
    const actionIndex = 2;
    const before = policyProbabilities(evaluatePolicy(inputs, weights))[actionIndex];

    const result = trainPolicyGradientBatch(weights, [{
      inputs,
      actionIndex,
      advantage: 1.4
    }], {
      learningRate: 0.14,
      l2: 0,
      gradientClip: 3
    });

    const after = policyProbabilities(evaluatePolicy(inputs, result.weights))[actionIndex];

    expect(result.loss).toBeGreaterThan(0);
    expect(after).toBeGreaterThan(before);
  });

  it('decreases negative-advantage action probability after a policy-gradient update', () => {
    const inputs = Array.from({ length: POLICY_INPUT_COUNT }, (_, index) =>
      index % 5 === 0 ? -0.45 : index % 5 === 1 ? 0.15 : index % 5 === 2 ? 0.4 : 0.05
    );
    const weights = createPolicyWeights(23);
    const actionIndex = 7;
    const before = policyProbabilities(evaluatePolicy(inputs, weights))[actionIndex];

    const result = trainPolicyGradientBatch(weights, [{
      inputs,
      actionIndex,
      advantage: -1.2
    }], {
      learningRate: 0.14,
      l2: 0,
      gradientClip: 3
    });

    const after = policyProbabilities(evaluatePolicy(inputs, result.weights))[actionIndex];

    expect(result.weights).toHaveLength(weights.length);
    expect(after).toBeLessThan(before);
  });

  it('clips policy-gradient updates that would move beyond the PPO trust region', () => {
    const inputs = Array.from({ length: POLICY_INPUT_COUNT }, (_, index) =>
      index % 2 === 0 ? 0.25 : -0.15
    );
    const weights = createPolicyWeights(29);
    const actionIndex = 3;
    const oldProbability = policyProbabilities(evaluatePolicy(inputs, weights))[actionIndex] * 0.2;

    const clipped = trainPolicyGradientBatch(weights, [{
      inputs,
      actionIndex,
      advantage: 1.5,
      oldProbability
    }], {
      learningRate: 0.2,
      l2: 0,
      gradientClip: 3,
      ppoClip: 0.2
    });

    expect(clipped.weights).toEqual(weights);
  });
});
