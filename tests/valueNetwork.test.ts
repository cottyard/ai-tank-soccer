import { describe, expect, it } from 'vitest';
import {
  createValueAdamMoments,
  createValueWeights,
  evaluateValue,
  trainValueBatch,
  valueLossGradients,
  valueWeightCount,
  type ValueSample
} from '../src/ai/valueNetwork';

const VALUE_INPUT_COUNT = 36;
const VALUE_WEIGHT_COUNT = valueWeightCount(VALUE_INPUT_COUNT);

function seededInputs(seed: number): number[] {
  let state = seed >>> 0;
  return Array.from({ length: VALUE_INPUT_COUNT }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state / 4294967296) * 2 - 1;
  });
}

describe('value network', () => {
  it('creates deterministic weights of the declared size', () => {
    const weights = createValueWeights(VALUE_INPUT_COUNT, 7);
    expect(weights).toHaveLength(VALUE_WEIGHT_COUNT);
    expect(createValueWeights(VALUE_INPUT_COUNT, 7)).toEqual(weights);
    expect(createValueWeights(VALUE_INPUT_COUNT, 8)).not.toEqual(weights);
  });

  it('produces a bounded scalar', () => {
    const weights = createValueWeights(VALUE_INPUT_COUNT, 3).map((value) => value * 40);
    for (let seed = 1; seed <= 24; seed += 1) {
      const value = evaluateValue(seededInputs(seed), weights);
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(-1);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it('rejects malformed inputs and weights', () => {
    const weights = createValueWeights(VALUE_INPUT_COUNT, 1);
    expect(() => evaluateValue([1, 2, 3], weights)).toThrow(/value weights for 3 inputs/);
    expect(() => evaluateValue(seededInputs(1), [1, 2, 3])).toThrow(/value weights for 36 inputs/);
    expect(() => evaluateValue([], weights)).toThrow(/must not be empty/);
  });

  it('matches a finite-difference gradient', () => {
    // A wrong backprop would still train to *something*, so check the analytic
    // gradient against numerical differentiation directly.
    const weights = createValueWeights(VALUE_INPUT_COUNT, 11);
    const samples: ValueSample[] = [
      { inputs: seededInputs(2), target: 0.7 },
      { inputs: seededInputs(3), target: -0.4 },
      { inputs: seededInputs(4), target: 0.1, weight: 2 }
    ];

    const { gradients } = valueLossGradients(samples, weights);
    const epsilon = 1e-6;
    const probeIndices = [0, 1, 37, 200, 900, VALUE_WEIGHT_COUNT - 1];

    for (const index of probeIndices) {
      const bumped = [...weights];
      bumped[index] += epsilon;
      const high = valueLossGradients(samples, bumped).loss;
      bumped[index] = weights[index] - epsilon;
      const low = valueLossGradients(samples, bumped).loss;
      const numeric = (high - low) / (2 * epsilon);
      expect(gradients[index]).toBeCloseTo(numeric, 6);
    }
  });

  it('fits a learnable signal and drives loss down', () => {
    // Target depends on a single input, so a correct trainer must reach a low
    // loss; a broken one plateaus near the variance of the targets.
    const samples: ValueSample[] = Array.from({ length: 64 }, (_unused, index) => {
      const inputs = seededInputs(index + 100);
      return { inputs, target: Math.tanh(inputs[0] * 2) };
    });

    let weights = createValueWeights(VALUE_INPUT_COUNT, 5);
    const moments = createValueAdamMoments(VALUE_INPUT_COUNT);
    const initialLoss = valueLossGradients(samples, weights).loss;

    let loss = initialLoss;
    for (let step = 0; step < 400; step += 1) {
      const result = trainValueBatch(samples, weights, { learningRate: 0.01, moments });
      weights = result.weights;
      loss = result.loss;
    }

    expect(loss).toBeLessThan(initialLoss * 0.25);
    expect(loss).toBeLessThan(0.02);
  });

  it('honours sample weights', () => {
    const inputs = seededInputs(21);
    const ignored: ValueSample[] = [{ inputs, target: 1, weight: 0 }];
    const weights = createValueWeights(VALUE_INPUT_COUNT, 9);
    const result = trainValueBatch(ignored, weights, { learningRate: 0.05 });
    expect(result.weights).toEqual([...weights]);
    expect(result.loss).toBe(0);
  });
});
