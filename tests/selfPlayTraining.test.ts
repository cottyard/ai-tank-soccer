import { describe, expect, it } from 'vitest';
import { createNeuralStrategy } from '../src/ai/neuralStrategy';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import {
  collectSelfPlaySamples,
  trainSelfPlayPolicy
} from '../src/ai/selfPlayTraining';
import { FIELD } from '../src/game/model';

describe('self-play neural training', () => {
  it('collects weighted winner-perspective samples from deterministic 1v1 rollouts', () => {
    const weights = defaultNeuralWeights();
    const result = collectSelfPlaySamples({
      weights,
      opponentWeights: weights,
      matches: 3,
      frames: 120,
      seed: 41
    });

    expect(result.samples.length).toBeGreaterThan(0);
    expect(result.frames).toBe(360);
    expect(result.redGoals).toBeGreaterThanOrEqual(0);
    expect(result.blueGoals).toBeGreaterThanOrEqual(0);
    expect(result.samples.every((sample) => sample.inputs.length === 36)).toBe(true);
    expect(result.samples.some((sample) => sample.weight > 1)).toBe(true);
  });

  it('uses deterministic short rollouts to label improved actions', () => {
    const weights = defaultNeuralWeights();
    const first = collectSelfPlaySamples({
      weights,
      opponentWeights: weights,
      matches: 2,
      frames: 90,
      rolloutFrames: 12,
      exploration: 0,
      seed: 73
    });
    const second = collectSelfPlaySamples({
      weights,
      opponentWeights: weights,
      matches: 2,
      frames: 90,
      rolloutFrames: 12,
      exploration: 0,
      seed: 73
    });

    expect(second.samples).toEqual(first.samples);
    expect(first.samples.length).toBeGreaterThan(0);
    expect(new Set(first.samples.map((sample) => sample.actionIndex)).size).toBeGreaterThan(1);
    expect(first.samples.some((sample) => sample.weight > 1.1)).toBe(true);
  });

  it('collects commands and labels for both self-play teams on each decision tick', () => {
    const weights = defaultNeuralWeights();
    const result = collectSelfPlaySamples({
      weights,
      opponentWeights: weights,
      matches: 1,
      frames: 6,
      rolloutFrames: 1,
      exploration: 0,
      seed: 19
    });
    const redSamples = result.samples.filter((sample) => sample.team === 'red');
    const blueSamples = result.samples.filter((sample) => sample.team === 'blue');

    expect(redSamples.length).toBeGreaterThan(0);
    expect(blueSamples.length).toBeGreaterThan(0);
    expect(redSamples.length).toBe(blueSamples.length);
  });

  it('seeds low-stamina and corner curriculum samples for policy improvement', () => {
    const weights = defaultNeuralWeights();
    const result = collectSelfPlaySamples({
      weights,
      opponentWeights: weights,
      matches: 8,
      frames: 48,
      rolloutFrames: 6,
      exploration: 0,
      seed: 23
    });

    expect(result.samples.some((sample) => sample.tags.includes('corner'))).toBe(true);
    expect(result.samples.some((sample) => sample.tags.includes('lowStamina'))).toBe(true);
    expect(result.samples.some((sample) => sample.inputs[7] < 0.5)).toBe(true);
  });

  it('seeds contested kickoff and own-corner fights for possession learning', () => {
    const weights = defaultNeuralWeights();
    const result = collectSelfPlaySamples({
      weights,
      opponentWeights: weights,
      matches: 10,
      frames: 48,
      rolloutFrames: 6,
      exploration: 0,
      seed: 31
    });
    const contestSamples = result.samples.filter((sample) => sample.tags.includes('contest'));
    const ownDangerCornerSamples = result.samples.filter(
      (sample) => sample.tags.includes('corner') && sample.tags.includes('ownDanger')
    );
    const activeContestSamples = contestSamples.filter((sample) => sample.actionIndex !== 4);

    expect(contestSamples.length).toBeGreaterThanOrEqual(8);
    expect(ownDangerCornerSamples.length).toBeGreaterThan(0);
    expect(activeContestSamples.length).toBeGreaterThan(contestSamples.length * 0.7);
    expect(Math.max(...contestSamples.map((sample) => sample.weight))).toBeGreaterThan(2.1);
  });

  it('does not over-label critically low-stamina samples as full-throttle pushes', () => {
    const weights = defaultNeuralWeights();
    const result = collectSelfPlaySamples({
      weights,
      opponentWeights: weights,
      matches: 10,
      frames: 72,
      rolloutFrames: 8,
      exploration: 0,
      seed: 37
    });
    const criticalSamples = result.samples.filter((sample) => sample.inputs[7] < 0.22);
    const fullThrottleCritical = criticalSamples.filter((sample) => sample.actionIndex === 0 || sample.actionIndex === 8);

    expect(criticalSamples.length).toBeGreaterThan(0);
    expect(fullThrottleCritical.length).toBeLessThan(criticalSamples.length * 0.35);
  });

  it('updates the policy from self-play without requiring human replay', () => {
    const weights = defaultNeuralWeights();
    const result = trainSelfPlayPolicy({
      weights,
      matches: 2,
      frames: 90,
      epochs: 4,
      batchSize: 16,
      learningRate: 0.02,
      seed: 5
    });

    expect(result.samples).toBeGreaterThan(0);
    expect(result.trainedSamples).toBeGreaterThan(0);
    expect(result.weights).toHaveLength(weights.length);
    expect(result.loss).toBeGreaterThanOrEqual(0);
    expect(result.modelVersion).toBe(1);
  });

  it('can use trained self-play weights in a normal neural strategy', () => {
    const trained = trainSelfPlayPolicy({
      weights: defaultNeuralWeights(),
      matches: 1,
      frames: 60,
      epochs: 2,
      batchSize: 8,
      seed: 12
    });
    const strategy = createNeuralStrategy({ weights: trained.weights });
    const state = trained.finalState;
    state.ball.position = { x: FIELD.length / 2, y: FIELD.width / 2 };

    const command = strategy.decide(state, 'red')['red-0'];

    expect([-1, 0, 1]).toContain(command.leftTrack);
    expect([-1, 0, 1]).toContain(command.rightTrack);
  });
});
