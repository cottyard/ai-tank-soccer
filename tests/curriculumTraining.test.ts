import { describe, expect, it } from 'vitest';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import { collectCurriculumSamples, trainCurriculumPolicy } from '../src/ai/curriculumTraining';

describe('curriculum policy training', () => {
  it('labels near-goal finishing scenarios as decisive forward pushes', () => {
    const result = collectCurriculumSamples({
      weights: defaultNeuralWeights(),
      scenarios: 80,
      rolloutFrames: 18,
      seed: 101
    });
    const finishSamples = result.samples.filter((sample) => sample.tags.includes('finish'));
    const forwardSamples = finishSamples.filter((sample) => sample.actionIndex === 8);

    expect(finishSamples.length).toBeGreaterThan(10);
    expect(forwardSamples.length / finishSamples.length).toBeGreaterThan(0.9);
  });

  it('generates balanced expert labels across tactical scenario families', () => {
    const result = collectCurriculumSamples({
      weights: defaultNeuralWeights(),
      scenarios: 24,
      rolloutFrames: 12,
      seed: 71
    });

    expect(result.samples.length).toBeGreaterThanOrEqual(24);
    expect(result.byScenario.finish).toBeGreaterThan(0);
    expect(result.byScenario.defense).toBeGreaterThan(0);
    expect(result.byScenario.corner).toBeGreaterThan(0);
    expect(result.byScenario.duel).toBeGreaterThan(0);
    expect(result.byScenario.kickoff).toBeGreaterThan(0);
    expect(result.byScenario.ownCornerContest).toBeGreaterThan(0);
    expect(new Set(result.samples.map((sample) => sample.actionIndex)).size).toBeGreaterThan(2);
    expect(result.samples.filter((sample) => sample.actionIndex === 8).length).toBeGreaterThan(4);
    expect(result.samples.filter((sample) => sample.actionIndex === 5).length).toBeLessThan(
      result.samples.length * 0.5
    );
    expect(result.samples.some((sample) => sample.tags.includes('finish'))).toBe(true);
    expect(result.samples.some((sample) => sample.tags.includes('ownDanger'))).toBe(true);
    expect(result.samples.some((sample) => sample.tags.includes('corner'))).toBe(true);
  });

  it('trains a policy from generated curriculum samples', () => {
    const weights = defaultNeuralWeights();
    const result = trainCurriculumPolicy({
      weights,
      scenarios: 32,
      rolloutFrames: 10,
      epochs: 3,
      batchSize: 16,
      learningRate: 0.006,
      seed: 83
    });

    expect(result.samples).toBeGreaterThanOrEqual(32);
    expect(result.trainedSamples).toBeGreaterThan(0);
    expect(result.loss).toBeGreaterThanOrEqual(0);
    expect(result.weights).toHaveLength(weights.length);
    expect(result.weights).not.toEqual(weights);
  });
});
