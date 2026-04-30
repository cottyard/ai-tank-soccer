import { describe, expect, it } from 'vitest';
import {
  collectPolicyGradientSelfPlay,
  trainPolicyGradientSelfPlay
} from '../src/ai/policyGradientTraining';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import { POLICY_INPUT_COUNT } from '../src/ai/policyNetwork';
import { FIELD, createInitialState, type GameState } from '../src/game/model';

describe('policy-gradient self-play training', () => {
  it('collects deterministic sampled decisions with policy-gradient fields', () => {
    const weights = defaultNeuralWeights();
    const first = collectPolicyGradientSelfPlay({
      weights,
      matches: 2,
      frames: 36,
      seed: 101,
      temperature: 0.95
    });
    const second = collectPolicyGradientSelfPlay({
      weights,
      matches: 2,
      frames: 36,
      seed: 101,
      temperature: 0.95
    });

    expect(second.decisions).toEqual(first.decisions);
    expect(first.frames).toBe(72);
    expect(first.decisions.length).toBeGreaterThan(0);
    expect(first.samples).toHaveLength(first.decisions.length);
    expect(first.decisions.every((decision) => decision.inputs.length === POLICY_INPUT_COUNT)).toBe(true);
    expect(first.decisions.every((decision) => decision.probability > 0 && decision.probability <= 1)).toBe(true);
    expect(first.decisions.every((decision) => Number.isFinite(decision.logProbability))).toBe(true);
  });

  it('assigns positive future return to the scoring side and negative return to the conceding side', () => {
    const weights = defaultNeuralWeights();
    const result = collectPolicyGradientSelfPlay({
      weights,
      matches: 1,
      frames: 12,
      seed: 3,
      discount: 1,
      goalReward: 1,
      winReward: 1,
      normalizeAdvantages: false,
      initialStateFactory: createImmediateRedGoalState
    });
    const openingRed = result.decisions.find((decision) => decision.team === 'red' && decision.frame === 0);
    const openingBlue = result.decisions.find((decision) => decision.team === 'blue' && decision.frame === 0);

    expect(result.redGoals).toBeGreaterThan(result.blueGoals);
    expect(openingRed?.return).toBeGreaterThan(0);
    expect(openingRed?.advantage).toBe(openingRed?.return);
    expect(openingBlue?.return).toBeLessThan(0);
  });

  it('can use outcome-curriculum starts to produce sparse reward without shaped tactical labels', () => {
    const weights = defaultNeuralWeights();
    const result = collectPolicyGradientSelfPlay({
      weights,
      matches: 6,
      frames: 18,
      seed: 13,
      discount: 1,
      goalReward: 1,
      winReward: 1,
      normalizeAdvantages: false,
      startStateMode: 'outcome-curriculum'
    });

    expect(result.redGoals + result.blueGoals).toBeGreaterThan(0);
    expect(result.decisions.some((decision) => decision.return !== 0)).toBe(true);
    expect(result.samples.some((sample) => sample.advantage !== 0)).toBe(true);
  });

  it('only trains the current policy side when playing against frozen opponent weights', () => {
    const weights = defaultNeuralWeights();
    const opponentWeights = weights.map((weight, index) => weight + (index % 2 === 0 ? 0.01 : -0.01));
    const result = collectPolicyGradientSelfPlay({
      weights,
      opponentWeights,
      matches: 2,
      frames: 36,
      seed: 17,
      temperature: 1
    });

    expect(result.decisions.length).toBeGreaterThan(0);
    expect(result.decisions.some((decision) => decision.team === 'red')).toBe(true);
    expect(result.decisions.some((decision) => decision.team === 'blue')).toBe(true);
    expect(result.samples).toHaveLength(result.decisions.length / 2);
    expect(result.samples).toEqual(
      result.decisions
        .filter((decision) => decision.trainable)
        .map((decision) => ({
          inputs: decision.inputs,
          actionIndex: decision.actionIndex,
          advantage: decision.advantage,
          oldProbability: decision.probability
        }))
    );
  });

  it('updates network weights from sparse self-play returns', () => {
    const weights = defaultNeuralWeights();
    const trained = trainPolicyGradientSelfPlay({
      weights,
      matches: 1,
      frames: 18,
      seed: 5,
      epochs: 2,
      batchSize: 4,
      learningRate: 0.02,
      discount: 1,
      normalizeAdvantages: false,
      initialStateFactory: createImmediateRedGoalState
    });
    const totalDelta = trained.weights.reduce(
      (sum, weight, index) => sum + Math.abs(weight - weights[index]),
      0
    );

    expect(trained.samples).toBeGreaterThan(0);
    expect(trained.weights).toHaveLength(weights.length);
    expect(trained.loss).not.toBe(0);
    expect(totalDelta).toBeGreaterThan(0);
  });
});

function createImmediateRedGoalState(): GameState {
  const state = createInitialState();
  state.ball.position = {
    x: FIELD.length - FIELD.ballRadius - 2,
    y: FIELD.width / 2
  };
  state.ball.velocity = { x: 220, y: 0 };
  return state;
}
